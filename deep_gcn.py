import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.models import DeepGCNLayer
from torch_geometric.nn import GATConv, GCNConv

    
class DeepGcn(nn.Module):
    def __init__(self, input_size, num_layers=1, hidden_size=64, dropout=0, lamda=1, alpha=0.1, variant=False):
        super(DeepGcn, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_layer = nn.Linear(input_size, hidden_size)   
        layers = []
        for _ in range(num_layers):
            conv = GCNConv(hidden_size, hidden_size)
            norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
            act = nn.ReLU()
            layers.append(DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout))
        self.gnn_model = nn.ModuleList(layers)
        self.pred_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x, adj):
        x = self.proj_layer(x)
        bs = x.size(0)
        out = []
        for i in range(bs):
            x_ = x[i]
            edge = adj[i].nonzero(as_tuple=False).t()
            for layer in self.gnn_model:
                x_ = layer(x_, edge)
            out.append(x_)
            
        out = torch.stack(out) # [bs, num_nodes, hidden_size]
        out = self.pred_head(out)
        return out