import torch 
import torch.nn as nn   

from .graph_conv import GCNConv
from .common import JumpingKnowledge
from .mlp import Mlp

class GraphMlp(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, jk='max'):
        super(GraphMlp, self).__init__()
        self.hidden_size = hidden_size            
        self.mlp = Mlp(input_size, hidden_size, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=True)
            
        
    def forward(self, x):
        # bs, n, m = x.size()
        # x = x.view(bs * n, m)
        mlp_out = self.mlp(x)
        # mlp_out = mlp_out.reshape(bs, n)
        return mlp_out
    