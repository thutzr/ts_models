# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/afi.py
"""AutomaticFeatureInteraction Model."""
import torch
from torch.nn import BatchNorm1d
import torch.nn as nn
from omegaconf import DictConfig

from pytorch_tabular.models.common.layers import Embedding2dLayer
from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

def _initialize_layers(activation, initialization, layers):
    if type(layers) is nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=(nn.init.calculate_gain(nonlinearity) if activation in ["ReLU", "LeakyReLU"] else 1),
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)

def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:

        layers.append(BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers



class Autoint(nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size=16,
                 num_layers=3,
                 deep_layers=False,
                 blocks_size=[128,64,32],
                 activation='relu',
                 initialization='kaiming',
                 use_bn=False,
                 dropout=0,
                 attn_dropouts=0,
                 attn_embed_dim=32,
                 num_heads=2,
                 num_attn_blocks=3,
                 attention_pooling=False,
                 has_residual=True,
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.deep_layers = deep_layers
        self.attention_pooling = attention_pooling
        self.has_residual = has_residual
        # Deep Layers
        _curr_units = hidden_size
        if num_layers:
            # Linear Layers
            layers = []
            curr_unit = hidden_size
            for i in range(num_layers):
                layers.extend(
                    _linear_dropout_bn(
                        activation,
                        initialization,
                        use_bn,
                        curr_unit,
                        blocks_size[i],
                        dropout,
                    )
                )
                curr_unit = blocks_size[i]
            self.linear_layers = nn.Sequential(*layers)
        # Projection to Multi-Headed Attention Dims
        self.attn_proj = nn.Linear(curr_unit, attn_embed_dim)
        _initialize_layers(activation, initialization, self.attn_proj)
        # Multi-Headed Attention Layers
        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    attn_embed_dim,
                    num_heads,
                    dropout=attn_dropouts,
                )
                for _ in range(num_attn_blocks)
            ]
        )
        if has_residual:
            self.V_res_embedding = torch.nn.Linear(
                curr_unit,
                (
                    attn_embed_dim * num_attn_blocks
                    if attention_pooling
                    else attn_embed_dim
                ),
            )
        self.output_dim = input_size * attn_embed_dim
        if attention_pooling:
            self.output_dim = self.output_dim * num_attn_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deep_layers:
            x = self.linear_layers(x)
        # (N, B, E*) --> E* is the Attn Dimention
        cross_term = self.attn_proj(x)
        if self.attention_pooling:
            attention_ops = []
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            if self.attention_pooling:
                attention_ops.append(cross_term)
        if self.attention_pooling:
            cross_term = torch.cat(attention_ops, dim=-1)
        # (B, N, E*)
        cross_term = cross_term.transpose(0, 1)
        if self.has_residuals:
            # (B, N, E*) --> Projecting Embedded input to Attention sub-space
            V_res = self.V_res_embedding(x)
            cross_term = cross_term + V_res
        # (B, NxE*)
        cross_term = nn.ReLU()(cross_term).reshape(-1, self.output_dim)
        return cross_term