from dataclasses import dataclass

@dataclass
class DIYCofig:
    """DIY 模型配置类，包含所有超参数。"""
    vocab_size: int = 6400
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    eps: float = 1e-5
    intermediate_size: int = 512