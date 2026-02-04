import math
import torch
from torch import nn

class BaseAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        print("x_max:", x_max)
        print("X_max_shape", x_max.shape)
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.hidden_size)

        att = self._softmax(scores, dim=-1)
        print("att:", att)

        return torch.matmul(att, V)
        
if __name__ == "__main__":
    baseattention = BaseAttention(2)

    x = torch.rand([1, 2, 2])
    print("x:" ,x)
    print("baseattention(x):" ,baseattention(x))