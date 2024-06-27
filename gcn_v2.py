# import torch.nn as nn
# import torch
# import math
# import numpy as np
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter

# class GraphConvolution(nn.Module):

#     def __init__(self, in_features, out_features, residual=False, variant=False):
#         super(GraphConvolution, self).__init__() 
#         self.variant = variant
#         if self.variant:
#             self.in_features = 2*in_features 
#         else:
#             self.in_features = in_features

#         self.out_features = out_features
#         self.residual = residual
#         self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj , h0 , lamda, alpha, l):
#         theta = math.log(lamda/l+1)
#         hi = torch.spmm(adj, input)
#         if self.variant:
#             support = torch.cat([hi,h0],1)
#             r = (1-alpha)*hi+alpha*h0
#         else:
#             support = (1-alpha)*hi+alpha*h0
#             r = support
#         output = theta*torch.mm(support, self.weight)+(1-theta)*r
#         if self.residual:
#             output = output+input
#         return output

# class GCNII(nn.Module):
#     def __init__(self, input_size, num_layers=1, hidden_size=64, output_size=64, dropout=0, lamda=1, alpha=0.1, variant=False):
#         super(GCNII, self).__init__()
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             self.convs.append(GraphConvolution(hidden_size, hidden_size,variant=variant))
#         self.fcs = nn.ModuleList()
#         self.fcs.append(nn.Linear(input_size, hidden_size))
#         self.fcs.append(nn.Linear(hidden_size, output_size))
#         self.params1 = list(self.convs.parameters())
#         self.params2 = list(self.fcs.parameters())
#         self.act_fn = nn.ReLU()
#         self.dropout = dropout
#         self.alpha = alpha
#         self.lamda = lamda

#     def forward(self, x, adj):
#         _layers = []
#         x = F.dropout(x, self.dropout, training=self.training)
#         layer_inner = self.act_fn(self.fcs[0](x))
#         _layers.append(layer_inner)
#         for i,con in enumerate(self.convs):
#             layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#             layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
#         layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#         layer_inner = self.fcs[-1](layer_inner)
#         return layer_inner
    
# class GcnV2(nn.Module):
#     def __init__(self, input_size, num_layers=1, hidden_size=64, dropout=0, lamda=1, alpha=0.1, variant=False):
#         super(GcnV2, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.gcn_model = GCNII(input_size, num_layers, hidden_size, dropout, lamda, alpha, variant)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size)
#         )
#         self.pred_head = nn.Linear(hidden_size+hidden_size, 1)
        
#     def forward(self, x, adj):
        
#         bs = x.size(0)
#         out = []
#         for i in range(bs):
#             adj[i] = adj[i] + torch.eye(adj[i].size(0)).to(adj.device)
#             D = torch.diag(torch.sum(adj[i], dim=1))
#             D_inv_root = torch.linalg.inv(torch.sqrt(D))
#             normalized_adj = torch.mm(torch.mm(D_inv_root, adj[i]), D_inv_root)
#             out.append(self.gcn_model(x[i], normalized_adj))
#         out = torch.stack(out) # [bs, num_nodes, hidden_size]
#         x = self.mlp(x)
#         out = torch.cat((out, x), dim=-1)
#         out = self.pred_head(out)
#         return out


import torch 
import torch.nn as nn   
from .graph_conv import GCN2Conv
from .mlp import Mlp

class GcnV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, normalize=True, bias=True, dropout=0, residual=False, variant=False, lamda=1, alpha=0.1):
        super(GcnV2, self).__init__()
        self.hidden_size = hidden_size
        self.embed_layer = nn.Linear(input_size, hidden_size)
        self.gcn_layers = nn.ModuleList([GCN2Conv(hidden_size, hidden_size, normalize=normalize, bias=bias, variant=variant, residual=residual, dropout=dropout) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        # self.pred_head = nn.Linear(hidden_size, output_size)
        self.pred_head = Mlp(hidden_size, hidden_size, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=True)
        self.dropout = nn.Dropout(dropout)
        self.lamda = lamda
        self.alpha = alpha 
        
        
    def forward(self, x, adj):
        x = self.embed_layer(x)
        x = self.dropout(x)
        h0 = x
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, adj, h0, self.lamda, self.alpha, i + 1)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.pred_head(x)
        return x 