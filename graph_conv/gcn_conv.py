import torch 
import torch.nn as nn 
import math

class GCNConv(nn.Module):
    def __init__(self, input_size, out_size, normalize=True, bias=True, dropout=0):
        super(GCNConv, self).__init__()
        
        self.input_size = input_size
        self.out_size = out_size
        self.normalize = normalize
        self.add_bias = bias
        self.droprate = dropout 
        if self.droprate > 0:
            self.dropout = nn.Dropout(self.droprate)
        
        self.theta = nn.Parameter(torch.FloatTensor(input_size, out_size))
        
        if self.add_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_size))
        
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.theta.data.uniform_(-stdv, stdv)
        if self.add_bias:
            self.bias.data.uniform_(-stdv, stdv)

        
    def forward(self, x, adj):
        '''
        x: [bs, num_nodes, input_size]
        adj: [bs, num_nodes, num_nodes]
        '''
        if self.normalize:
            adj = self.add_self_loop(adj)
            adj = self.normalize_adj(adj)
            
        support = torch.matmul(x, self.theta) # [bs, num_nodes, out_size]
        out = torch.bmm(adj, support)
        if self.add_bias:
            out = out + self.bias
        if self.droprate > 0:
            out = self.dropout(out)
        return out
        
    def add_self_loop(self, adj):
        # adj : [bs, num_nodes, num_nodes]
        adj = adj + torch.eye(adj.size(1)).to(adj.device)
        return adj 
        
    def normalize_adj(self, adj):
        # adj: [bs, num_nodes, num_nodes]
        degree = torch.diag_embed(torch.sum(adj, dim=2))
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree))
        adj_normalized = torch.bmm(torch.bmm(degree_inv_sqrt, adj), degree_inv_sqrt)
        return adj_normalized