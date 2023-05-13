import torch
import torch.nn as nn


class DDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ddae = nn.Sequential(
            nn.Linear(257, 512, bias=True),
            nn.LeakyReLU(),
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(),
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(),
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(),
        )
        self.last_layer = nn.Linear(512, 257, bias=False)

    def forward(self, x):
        z = self.ddae(x)
        y = self.last_layer(z)
        return z, y


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=257, hidden_size=300, batch_first=True, num_layers=1, bidirectional=False)
        self.last_layer = nn.Linear(300, 257, bias=False)

    def forward(self, x):
        z, _ = self.lstm(x)
        y = self.last_layer(z)
        return z, y


class BLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.blstm = nn.LSTM(input_size=257, hidden_size=300, batch_first=True, num_layers=1, dropout=0.0, bidirectional=True)
        self.last_layer = nn.Linear(300, 257, bias=False)
    
    def forward(self, x):
        h, _ = self.blstm(x)
        z = h[:, :, :int(h.size(-1) / 2)] + h[:, :, int(h.size(-1) / 2):]
        y = self.last_layer(z)
        return z, y
