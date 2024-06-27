import torch 
import torch.nn as nn   

from .graph_conv import GCNConv
from .common import JumpingKnowledge
from .mlp import Mlp

class JumpGcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, jk='max'):
        super(JumpGcn, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(input_size, hidden_size)
        layers = []
        for i in range(num_layers):
            layers.append(GCNConv(hidden_size, hidden_size, dropout=dropout))
            
        self.gnn_model = nn.ModuleList(layers)
        
        self.pred_head = nn.Linear(hidden_size, 1)
        
        self.jk = JumpingKnowledge(jk, hidden_size, num_layers)
        if jk == 'cat':
            in_channels = hidden_size * num_layers
        else:
            in_channels = hidden_size
            
        # self.pred_head = Mlp(in_channels, hidden_size, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=True)
        self.mlp = Mlp(input_size, hidden_size, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=True)
        self.pred_head = nn.Linear(in_channels, 1)
            
        
    def forward(self, x, adj):
        mlp_out = self.mlp(x)
        x = self.proj_layer(x)
        xs = []
        for i, layer in enumerate(self.gnn_model):
            x = layer(x, adj)
            x = torch.relu(x)
            xs.append(x)
            
        out = self.jk(xs)
        out = self.pred_head(out)
        return out / 2 + mlp_out / 2
    