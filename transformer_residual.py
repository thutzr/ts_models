import torch
from torch import nn
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('position encoder', self.pe.shape)
        return x + self.pe[:, :x.size(1), :]   

class Transformer(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.pos_encoder = PositionalEncoding(hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        # x: [N, T, F]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x.squeeze()
    
    
class TransformerResidual(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0, seq_len=10):
        super().__init__()

        # self.src_mask = None 
        # self.proj_layer = nn.Linear(input_size, hidden_size)
        # self.tanh = nn.Tanh()
        self.pos_encoder = PositionalEncoding(input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=6, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.decoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )
        # self.linear = nn.Linear(10, 1)
        self.decoder_pool = nn.Linear(seq_len, 1)
        self.mlp_head = nn.Linear(hidden_size, 1)
        # self.fc = nn.Linear(hidden_size, 1)
        # self.input_size = input_size

    def forward(self, x):
        
        # x: [N, T, F]
        # x = self.proj_layer(x)
        # x = self.tanh(x)
        out = self.pos_encoder(x)
        out = self.transformer_encoder(out)
        out, _ = self.decoder(out)
        out = self.decoder_pool(torch.permute(out, (0, 2, 1))).squeeze()
        out = self.mlp_head(out)
        return out.squeeze()