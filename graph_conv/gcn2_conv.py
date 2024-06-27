import torch 
import torch.nn as nn 
import math 

class GCN2Conv(nn.Module):
    def __init__(self, input_size, out_size, normalize=True, bias=True, dropout=0, residual=False, variant=False):
        super(GCN2Conv, self).__init__()
        self.variant = variant
        if variant:
            self.input_size = input_size * 2
        else:
            self.input_size = input_size
        self.out_size = out_size
        self.residual = residual 
        self.normalize = normalize
        self.add_bias = bias
        self.droprate = dropout 
        if self.droprate > 0:
            self.dropout = nn.Dropout(self.droprate)
        
        self.theta = nn.Parameter(torch.FloatTensor(self.input_size, out_size))
        
        if self.add_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_size))
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.theta.data.uniform_(-stdv, stdv)
        if self.add_bias:
            self.bias.data.uniform_(-stdv, stdv)
        
        
    def forward(self, x, adj, h0 , lamda, alpha, l):
        if self.normalize:
            adj = self.add_self_loop(adj)
            adj = self.normalize_adj(adj)
            
        beta = math.log(lamda/l+1)
        hi = torch.bmm(adj, x) # [bs, num_nodes, input_size]
        
        if self.variant:
            support = torch.cat([hi, h0], 1) # [bs, num_nodes, input_size*2]
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
            
        output = beta * torch.matmul(support, self.theta) + (1 - beta) * r
        if self.residual:
            output = output + x
        return output
        
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