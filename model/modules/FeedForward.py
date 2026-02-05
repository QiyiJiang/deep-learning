import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(self.hidden_size, 2*self.hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class GatedFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.linear_gate = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear_up = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear_down = nn.Linear(self.intermediate_size, self.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.linear_gate(x)
        gate = self.activation(gate)
        gate = gate * self.linear_up(x)
        gate = self.linear_down(gate)
        return self.dropout(gate)

