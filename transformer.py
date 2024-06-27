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
        return x + self.pe[:, :x.size(1), :] * 0.05

# class Transformer(nn.Module):
#     def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0):
#         super().__init__()

#         self.pos_encoder = PositionalEncoding(hidden_size)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.decoder = nn.Linear(hidden_size, 1)

#     def forward(self, x):
        
#         # x: [N, T, F]
#         x = self.pos_encoder(x)
#         x = self.transformer_encoder(x)
#         x = self.decoder(x[:, -1, :])
#         return x.squeeze()
    
    
class Transformer(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        # self.src_mask = None 
        self.feature_embedder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),            
        )
        # self.tanh = nn.Tanh()
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=256, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_attn = nn.MultiheadAttention(embed_dim=hidden_size, 
                                              num_heads=1, 
                                              batch_first=True)
        self.decoder_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.decoder_norm1 = nn.LayerNorm(hidden_size)
        self.decoder_norm2 = nn.LayerNorm(hidden_size)
        self.decoder_fc1 = nn.Linear(hidden_size * 3, 1)
        # self.decoder_fc2 = nn.Linear(hidden_size, 1)
        # self.relu = nn.ReLU()
        # self.linear = nn.Linear(10, 1)
        

        # self.fc = nn.Linear(hidden_size, 1)
        # self.input_size = input_size

    def forward(self, x):
        
        # x: [N, T, F]
        # x = self.proj_layer(x)
        x = self.feature_embedder(x)
        # x = self.tanh(x)
        out = self.pos_encoder(x)
        out = self.transformer_encoder(out)
        decoder_out, _ = self.decoder_attn(out[:, -1:, :], out, out)
        decoder_out = self.decoder_norm1(decoder_out + out[:, -1:, :])
        decoder_ff_out = self.decoder_ff(decoder_out[:, -1:, :])
        decoder_out = self.decoder_norm2(decoder_ff_out + decoder_out[:, -1:, :])
        out = self.decoder_fc1(torch.cat((x[:, -1, :], out[:, -1, :], decoder_out[:, -1, :]), dim=1))
        return out.squeeze()