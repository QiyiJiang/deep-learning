import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x)

if __name__ == "__main__":
    resnorm = RMSNorm(2)
    x = torch.rand([1,2,2])
    print(x)
    print(resnorm(x))
