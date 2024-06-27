from typing import Optional

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class DeepGNNLayer(nn.Module):
    
    def __init__(
        self,
        conv: Optional[Module] = None,
        norm: Optional[Module] = None,
        act: Optional[Module] = None,
        block: str = 'res+',
        dropout: float = 0.,
    ):
        super().__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        # self.norm.reset_parameters()

    def forward(self, x, adj, *args, **kwargs) -> Tensor:

        if self.block == 'res+':
            h = x
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv(h, adj, *args, **kwargs)

            return x + h

        else:
            h = self.conv(x, adj, *args, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == 'res':
                h = x + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass

            return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'