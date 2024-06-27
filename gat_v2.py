import torch 
import torch.nn as nn   

# from torch_geometric.nn import GATv2Conv
# from torch_geometric.nn.models import GAT 
from .gat import GAT
# class GatV2(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, heads=1, dropout=0):
#         super(GatV2, self).__init__()
#         self.hidden_size = hidden_size
#         self.proj_layer = nn.Linear(input_size, hidden_size)
#         self.convs = nn.ModuleList(
#             [GATv2Conv(hidden_size, hidden_size // heads, heads=heads, dropout=dropout) for _ in range(num_layers)] 
#         )
#         self.act = nn.ReLU()
        
#         self.linear = nn.Linear(hidden_size, 1)
        
#     def forward(self, x, adj):
#         x = self.proj_layer(x)
#         # get edge index (shape: [2, num_edges]) from an adjacency matrix
#         bs = x.size(0)
#         out = []
#         for i in range(bs):
#             x_ = x[i]
#             adj_ = adj[i]
#             out.append(self._forward(x_, adj_))
            
#         out = torch.stack(out, dim=0)
#         return out
            
#     def _forward(self, x, adj):
#         adj = self.add_self_loops(adj)
#         edge_index = adj.nonzero(as_tuple=False).t()
#         for conv in self.convs:
#             x = self.act(conv(x, edge_index))
#         x = self.linear(x)
#         return x
    
#     def add_self_loops(self, adj):
#         adj = adj + torch.eye(adj.size(0)).to(adj.device)
#         return adj


# class GatV2(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, heads=1, dropout=0):
#         super(GatV2, self).__init__()
#         self.hidden_size = hidden_size
#         self.proj_layer = nn.Linear(input_size, hidden_size)
#         self.gnn_model = GAT(hidden_size, hidden_size*2, num_layers, hidden_size, v2=True, dropout=dropout)
        
#         self.linear = nn.Linear(hidden_size, 1)
        
#     def forward(self, x, adj):
#         x = self.proj_layer(x)
#         bs = x.size(0)
#         out = []
#         for i in range(bs):
#             out.append(self._forward(x[i], adj[i]))
            
#         out = torch.stack(out, dim=0)
#         out = self.linear(out)
#         return out
            
#     def _forward(self, x, adj):
#         edge_index = adj.nonzero(as_tuple=False).t()
#         x = self.gnn_model(x, edge_index)
#         return x


class GatV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, heads=1, dropout=0):
        super(GatV2, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(input_size, hidden_size)
        # self.gnn_model = GAT(hidden_size, hidden_size*2, num_layers, hidden_size, v2=False, dropout=dropout)
        self.gnn_model = GAT(num_of_layers=num_layers, num_heads_per_layer=[heads]*num_layers, num_features_per_layer=[hidden_size]*(num_layers + 1), add_skip_connection=True, bias=True, dropout=dropout, log_attention_weights=False, v2=True)
        self.pred_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x, adj):
        x = self.proj_layer(x)
        out = self.gnn_model(x, adj)
        out = self.pred_head(out)
        return out