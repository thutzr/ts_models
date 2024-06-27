import torch
from torch import nn

class Lstm(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        # self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        # self.fc2 = nn.Linear(hidden_size // 2, 1)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1)
        self.input_size = input_size

    def forward(self, x):
        
        # x: [N, T, F]
        out, _ = self.rnn(x) # out: [N, T, H]
        last_out = out[:, -1, :].squeeze() # [N, H]
        # out = self.fc1(self.bn1(last_out))
        # out = self.relu(out)
        # out = self.fc2(self.bn2(out))
        out = self.fc(last_out)
        return out 