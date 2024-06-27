import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .crossformer_encoder import Encoder
from .crossformer_decoder import Decoder
from .crossformer_attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from .crossformer_embed import DSW_embedding

from math import ceil



class Crossformer(nn.Module):
    def __init__(self, input_size, seq_len, seg_len=2, out_len=1, win_size = 4,
                factor=1, d_model=64, d_ff = 256, n_heads=4, num_layers=3, 
                dropout=0.0, baseline = False):
        super(Crossformer, self).__init__()
        self.data_dim = input_size
        self.in_len = seq_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * seq_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, input_size, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(num_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, input_size, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, num_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
        self.final_fc = nn.Linear(input_size, 1)
        
    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        final_pred_y = base + predict_y[:, :self.out_len, :]
        final_pred = self.final_fc(final_pred_y)
        return final_pred.squeeze()