import torch
from torch import nn
from model.modules.modelconfig import DIYCofig

class RMSNorm(nn.Module):
    """RMS Normalization：对最后一维做 RMS 归一化。"""
    
    def __init__(self, config: DIYCofig) -> None:
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS 归一化：x / sqrt(mean(x²) + eps)。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: Input tensor，shape (..., hidden_size)
        
        Returns:
            Normalized tensor，shape (..., hidden_size)
        """
        return self.weight * self._norm(x)

if __name__ == "__main__":
    resnorm = RMSNorm(2)
    x = torch.rand([1,2,2])
    print(x)
    print(resnorm(x))
