import torch
from torch import nn
from .mlp import Mlp
import math
from .itransformer_common import RevIN


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # print('position encoder', self.pe.shape)
#         return x + self.pe[:, :x.size(1), :]   

class Alstm2(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, seq_len=10, dropout=0.0, rnn_type="GRU",use_reversible_instance_norm=False, reversible_instance_norm_affine=False):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = input_size
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self.seq_len = seq_len
        # self.reverse = RevIN(self.input_size, affine = reversible_instance_norm_affine) if use_reversible_instance_norm else None
        # self.instance_norm = nn.InstanceNorm1d(self.input_size, affine = True)
        self._build_model()

    def _build_model(self):
        
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Linear(in_features=self.input_size, out_features=self.hid_size)
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        
        self.att_net = nn.MultiheadAttention(embed_dim=self.hid_size, 
                                              num_heads=2, 
                                              batch_first=True)
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        
        # self.feature_att = nn.MultiheadAttention(embed_dim=self.seq_len, 
        #                                       num_heads=1, 
        #                                       batch_first=True)
        # self.pred_head = nn.Linear(in_features=self.hid_size * 3, out_features=1)
        

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, input_size]
        # inputs = self.instance_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        inputs = self.net(inputs)
        rnn_out, _ = self.rnn(inputs)  # [batch, seq_len, num_directions * hidden_size]
        out_att, _ = self.att_net(rnn_out, rnn_out, rnn_out)  # [batch, seq_len, 1]
            
        out = torch.cat((rnn_out[:, -1, :], out_att[:, -1, :]), dim=1)
        out = self.fc_out(out).squeeze() # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        
        return out