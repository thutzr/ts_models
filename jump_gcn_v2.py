import torch 
import torch.nn as nn   

from .graph_conv import GCN2Conv
from .common import JumpingKnowledge
from .mlp import Mlp

class JumpGcnV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, jk='max', residual=False, variant=False, lamda=1, alpha=0.1):
        super(JumpGcnV2, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(input_size, hidden_size)
        layers = []
        for i in range(num_layers):
            layers.append(GCN2Conv(hidden_size, hidden_size, dropout=dropout, residual=residual, variant=variant))
            
        self.gnn_model = nn.ModuleList(layers)
        self.pred_head = nn.Linear(hidden_size, 1)
        
        self.jk = JumpingKnowledge(jk, hidden_size, num_layers)
        if jk == 'cat':
            in_channels = hidden_size * num_layers
        else:
            in_channels = hidden_size
            
        self.mlp = Mlp(input_size, hidden_size, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=True)
        self.pred_head = nn.Linear(in_channels, 1)
        self.lamda = lamda 
        self.alpha = alpha 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        mlp_out = self.mlp(x)
        x = self.proj_layer(x)
        xs = []
        x = self.dropout(x)
        h0 = x.clone()
        for i, layer in enumerate(self.gnn_model):
            x = layer(x, adj, h0, lamda=self.lamda, alpha=self.alpha, l=i+1)
            x = self.relu(x)
            x = self.dropout(x)
            xs.append(x)
            
        out = self.jk(xs)
        out = self.pred_head(out)
        return out / 2 + mlp_out / 2
    