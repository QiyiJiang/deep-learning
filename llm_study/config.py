from dataclasses import dataclass
import torch


@dataclass
class DIYConfig:
    """DIY 模型配置类，包含所有超参数。"""
    vocab_size: int = 6400
    hidden_size: int = 1024
    num_layers: int = 29
    num_heads: int = 16
    intermediate_size: int = 4096
    max_seq_len: int = 2048
    dropout: float = 0.1
    eps: float = 1e-5
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
