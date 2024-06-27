import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.models import PNA

    
class Pna(nn.Module):
    def __init__(self, input_size, num_layers=1, hidden_size=64, dropout=0, lamda=1, alpha=0.1, variant=False):
        super(Pna, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_layer = nn.Linear(input_size, hidden_size)   
        self.gnn_model = PNA(hidden_size, hidden_size*2, num_layers, hidden_size, dropout)
        
        self.pred_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x, adj):
        x = self.proj_layer(x)
        bs = x.size(0)
        out = []
        for i in range(bs):
            adj[i] = adj[i] + torch.eye(adj[i].size(0)).to(adj.device)
            edge = adj[i].nonzero(as_tuple=False).t()
            out.append(self.gnn_model(x[i], edge))
            
        out = torch.stack(out) # [bs, num_nodes, hidden_size]
        out = self.pred_head(out)
        return out