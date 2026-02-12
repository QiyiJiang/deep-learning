from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class DIYConfig:
    """DIY 模型配置类，包含模型结构、数据与示例脚本的默认超参数。"""

    # 模型结构（默认 ~100M 参数：512 hidden, 23 layers, 8 heads, 2048 FFN）
    vocab_size: int = 6400
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_key_value_heads: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.1
    eps: float = 1e-5
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # RoPE
    rope_base: float = 1e6

    # 数据管道默认（datasets 未显式传 max_length 时使用，应 <= max_seq_len）
    train_max_length: int = 340

    # 示例脚本默认训练配置（examples 中 argparse 未指定时使用）
    default_lr: float = 5e-4
    default_accumulation_steps: int = 8 
    default_batch_size: int = 32
    default_save_step: int = 100000
    default_pretrain_data_path: str = "dataset/pretrain_hq.jsonl"
    default_sft_data_path: str = "dataset/sft_mini_512_15.jsonl"
    default_checkpoints_dir: str = "checkpoints"
    default_num_epochs: int = 3
    default_demo_batch_size: int = 2
    default_demo_seq_len: int = 10
