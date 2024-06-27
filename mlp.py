import torch
from torch import nn

class Mlp(nn.Module):

    def __init__(self, input_size, hidden_size=512, num_layers=3, dropout=0.0, activation='relu', use_bn=False, use_ln=False):
        super().__init__()
        
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size + input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        
        self.config_activation(activation)
        
        if use_bn:
            self.bn1 = nn.BatchNorm1d(input_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
        if use_ln:
            self.ln1 = nn.LayerNorm(input_size)
            self.ln2 = nn.LayerNorm(hidden_size + input_size)
            self.ln3 = nn.LayerNorm(hidden_size)
            
        self.use_bn = use_bn
        self.use_ln = use_ln
        
        # self.first_pred_head = nn.Linear(hidden_size, 1)
        # self.second_pred_head = nn.Linear(hidden_size, 1)
        # self.third_pred_head = nn.Linear(hidden_size, 1)
        
    def config_activation(self, activation):
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Invalid Activation Type {activation}!")

    def forward(self, x):
        # [N, F]
        # return self.mlp(x).squeeze()
        if self.use_ln:
            x = self.ln1(x)
        out = self.fc1(x)
        out = self.act(out)
        
        if self.use_bn:
            out = self.bn1(out)
        out = self.fc2(out)
        out = self.act(out)
        if self.use_bn:
            out = self.bn2(out)
        out = torch.cat((out, x), dim=-1)
        x = self.fc3(out)
        x = self.act(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.fc4(x)
        return x