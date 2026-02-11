import torch
from torch import nn
from .config import DIYConfig


class FeedForward(nn.Module):
    """标准 FeedForward 层：Linear → GELU → Dropout → Linear → Dropout。"""
    
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(self.hidden_size, 2*self.hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: Input tensor，shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor，shape (batch_size, seq_len, hidden_size)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class GatedFeedForward(nn.Module):
    """Gated FeedForward 层：gate(x) * up(x) → down，使用 GELU 激活。"""
    
    def __init__(self, config: DIYConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = 64 * ((int(config.hidden_size * 8 / 3) + 64 - 1) // 64)

        self.linear_gate = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear_up = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear_down = nn.Linear(self.intermediate_size, self.hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: Input tensor，shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor，shape (batch_size, seq_len, hidden_size)
        """
        gate = self.linear_gate(x)
        gate = self.activation(gate)
        gate = gate * self.linear_up(x)
        gate = self.linear_down(gate)
        return self.dropout(gate)
