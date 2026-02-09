"""llm_study: 独立学习用 DIY 小 LLM 实现（配置、模型、数据集、RoPE）。"""

from .config import DIYConfig
from .model import DIYModel, DIYForCausalLM
from .datasets import (
    PretrainDataset,
    SimplePretrainDataset,
    SimpleSFTDataset,
    SFTDataset,
)
from .rope import precompute_freqs_cis, apply_rotary_pos_emb
from .logger import (
    get_logger,
    setup_logger,
    debug,
    info,
    warning,
    error,
    critical,
)

__all__ = [
    "DIYConfig",
    "DIYModel",
    "DIYForCausalLM",
    "PretrainDataset",
    "SimplePretrainDataset",
    "SimpleSFTDataset",
    "SFTDataset",
    "precompute_freqs_cis",
    "apply_rotary_pos_emb",
    "get_logger",
    "setup_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]
