import torch 
import torch.nn as nn   
from .graph_conv import GCNConv
from .mlp import Mlp

class Gcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(Gcn, self).__init__()
        self.hidden_size = hidden_size
        self.embed_layer = nn.Linear(input_size, hidden_size)
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        # self.pred_head = nn.Linear(hidden_size, output_size)
        self.pred_head = Mlp(hidden_size, hidden_size, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=True)
        
        
    def forward(self, x, adj):
        x = self.embed_layer(x)
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.relu(x)
        
        x = self.pred_head(x)
        return x 