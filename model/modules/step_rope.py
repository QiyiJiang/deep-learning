import torch
import math
from torch import nn

def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6):
    """预计算 RoPE 的 cos/sin 值。
    
    Args:
        dim: head_dim（每个头的维度）
        end: 最大位置（max_seq_len）
        rope_base: 基础频率，默认 1e6
    
    Returns:
        freqs_cos: cos 值，shape (end, dim)
        freqs_sin: sin 值，shape (end, dim)
    """
    # 计算基础频率：只取偶数索引（0, 2, 4, ...）
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算所有位置的频率矩阵
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # shape: (end, dim//2)
    
    # 生成 cos/sin
    freqs_cos = torch.cos(freqs)  # shape: (end, dim//2)
    freqs_sin = torch.sin(freqs)  # shape: (end, dim//2)
    
    # 重复一次以匹配完整维度
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)  # shape: (end, dim)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)  # shape: (end, dim)
    
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """对 Q/K 应用旋转位置编码。
    
    Args:
        q: Query，shape (batch, num_heads, seq_len, head_dim)
        k: Key，shape (batch, num_heads, seq_len, head_dim)
        cos: cos 值，shape (seq_len, head_dim)
        sin: sin 值，shape (seq_len, head_dim)
    
    Returns:
        q_rot: 旋转后的 Query
        k_rot: 旋转后的 Key
    """
    def rotate_half(x):
        """把后一半维度取负并与前一半拼接。"""
        d = x.shape[-1]
        return torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)
    
    # q, k: (B, H, L, D)
    # cos, sin: (L, D)

    # 扩展到 (1, 1, L, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot