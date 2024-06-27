import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from collections import namedtuple
from packaging import version
from functools import wraps


EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        heads = None,
        scale = None,
        flash = False,
        causal = False
    ):
        super().__init__()
        self.scale = scale

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal

        # flash attention

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        major, minor = device_properties.major, device_properties.minor

        if (major, minor) == (8, 0):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        elif (major, minor) == (9, 0):
            print_once('H100 GPU detected, using flash attention')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def flash_attn(
        self,
        q, k, v
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal = self.causal,
                dropout_p = self.dropout if self.training else 0.
            )
        
        return out

    def forward(
        self,
        q, k, v
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device, dtype = q.shape[-2], q.shape[1], k.shape[1], q.device, q.dtype

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale

        if self.causal:
            i, j, dtype = *sim.shape[-2:], sim.dtype
            mask_value = -torch.finfo(sim.dtype).max
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = attn.type(dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)

        return out
    
    
Statistics = namedtuple('Statistics', [
    'mean',
    'variance',
    'gamma',
    'beta'
])

class RevIN(Module):
    def __init__(
        self,
        num_variates,
        affine = True,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1), requires_grad = affine)
        self.beta = nn.Parameter(torch.zeros(num_variates, 1), requires_grad = affine)

    def forward(self, x, return_statistics = False):
        # x is of shape (batch, num_variates, sequence)
        assert x.shape[1] == self.num_variates
        # mean and variance are computed over the sequence dimension
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        var_rsqrt = var.clamp(min = self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min = self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        statistics = Statistics(mean, var, self.gamma, self.beta)

        return rescaled, reverse_fn, statistics
