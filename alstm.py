import torch
from torch import nn
from .mlp import Mlp
import math
from .itransformer_common import RevIN
from .common import sLSTM, mLSTM
from .attn import FlowAttention, FullAttention, AttentionLayer, ProbAttention
from .dilated_rnn import DilatedRnn
from einops import rearrange

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

class Alstm(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, seq_len=10, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = input_size
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self.seq_len = seq_len
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
        # self.rnn = DilatedRnn(
        #     input_size=self.hid_size,
        #     hidden_size=self.hid_size,
        #     num_layers=self.rnn_layer,
        #     batch_first=True,
        #     dropout=self.dropout,
        # )
        self.att_net = nn.MultiheadAttention(embed_dim=self.hid_size, 
                                              num_heads=1, 
                                              batch_first=True)
        # self.att_net = AttentionLayer(
        #     FullAttention(), d_model=self.hid_size, n_heads=1,
        # )
        self.time_fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        
        # self.norm2 = nn.LayerNorm(self.hid_size)
        # self.feature_att_net = nn.MultiheadAttention(embed_dim=self.seq_len, 
        #                                       num_heads=1, 
        #                                       batch_first=True)
        # self.feature_fc_out = nn.Linear(in_features=self.hid_size, out_features=1)
        

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, input_size]

        inputs = self.net(inputs)
        # inputs = rearrange(inputs, 'b (n s) f -> (n b) s f', n=2)
        rnn_out, _ = self.rnn(inputs)  # [batch, seq_len, num_directions * hidden_size]
        # rnn_out = rearrange(rnn_out, ' (n b) s f -> b (n s) f', n=2)
        # 加入时间信息position embedding
        out_att, _ = self.att_net(rnn_out, rnn_out, rnn_out)  # [batch, seq_len, 1]
        out = torch.cat((rnn_out[:, -1, :], out_att[:, -1, :]), dim=1)
        out = self.time_fc_out(out)
        
        return out