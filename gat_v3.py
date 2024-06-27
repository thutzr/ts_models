import torch 
import torch.nn as nn   


class GatV3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, heads=1, output_size=1):
        super(GatV3, self).__init__()
        self.hidden_size = hidden_size
        self.embed_layer = nn.Linear(input_size, hidden_size)
        self.heads = heads
        self.node_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, heads, dim_feedforward=hidden_size*4, dropout=0, batch_first=True), num_layers=num_layers
        )
        self.relu = nn.ReLU()
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        
    def forward(self, x, adj):
        x = self.embed_layer(x)
        bs, n, n = adj.size()
        adj = adj + torch.eye(n).to(adj.device)
        mask = adj.unsqueeze(1).repeat(1, self.heads, 1, 1).view(bs*self.heads, n, n)
        mask = (1 - mask).bool()
        x = self.node_attn(x, mask=mask)
        x = self.linear(x)
        return x 
