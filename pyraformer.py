import torch 
import torch.nn as nn
from pyraformer.Pyraformer_LR import Model 
from argparse import Namespace

class Pyraformer(nn.Module):
    def __init__(self, input_size, window_size='[4,4,4]', inner_size=3, heads=4, d_model=512, d_inner_hid=512, d_k=128, d_v=128, num_layers=4, dropout=0.1, seq_len=10, covariate_size=0, use_tvm=False, d_bottleneck=128, truncate=False):
        super().__init__()
        opt = Namespace()
        opt.input_size = input_size
        opt.window_size = eval(window_size)
        opt.inner_size = inner_size
        opt.n_head = heads
        opt.d_model = d_model
        opt.d_inner_hid = d_inner_hid
        opt.d_k = d_k
        opt.d_v = d_v
        opt.n_layer = num_layers
        opt.dropout = dropout
        opt.num_seq = seq_len
        opt.covariate_size = covariate_size
        opt.use_tvm = use_tvm
        opt.d_bottleneck = d_bottleneck
        opt.truncate = truncate
        self.model = Model(opt)
        
        
    def forward(self, x):
        return self.model(x)
