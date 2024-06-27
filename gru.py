import torch
from torch import nn

class Gru(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        # self.fc2 = nn.Linear(hidden_size // 2, 1)
        # self.layer_norm = nn.LayerNorm(hidden_size // 2)
        # self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, 1)
        # self.bn = nn.BatchNorm1d(hidden_size)

        self.input_size = input_size

    def forward(self, x):
        # x: [N, T, F]
        out, _ = self.rnn(x) # out: [N, T, H]
        # out = out[:, -1, :].squeeze()
        # out = self.bn(out)
        # out = self.fc1(out)
        # out = self.layer_norm(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.fc_out(out)
        # return out 
        return self.fc_out(out[:, -1, :]).squeeze()