import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .config import DIYConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将KV头重复n_rep次以匹配Q头数量"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class BaseAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.hidden_size)

        att = self._softmax(scores, dim=-1)

        return torch.matmul(att, v)

"""
MaskedAttention 主要解决自回归建模中信息泄露的问题
1. causal mask 防止看到未来
2. padding mask 防止模型关注无意义的 padding token
3. combined mask 混合 mask
"""
class MaskedAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):

        _, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.hidden_size)
        print("scores:", scores)
        print("scores shape:", scores.shape)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float("-inf"))
        print("scores:", scores)
        print("scores shape:", scores.shape)

        att = self._softmax(scores, dim=-1)

        return torch.matmul(att, v)


"""
把 hidden space 切成多个子空间，每个子空间独立做 attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        """
        NOTE .contiguous() 可以将逻辑上正确但是内存不连续的 Tensor 转换为连续的，因为 view 要求 Tensor 内存连续
        """
        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att

class PaddingMaskedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x, seq_lengths=None):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]
        if seq_lengths is not None:
            # padding mask
            padding_mask = torch.arange(seq_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
            padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
            final_mask = causal_mask | padding_mask
        else:
            final_mask = causal_mask

        scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att


"""
在训练阶段：
输入整个序列 x = [x1, x2, ..., xL]
Multi-head attention 会同时计算每个 query 对所有 key 的 attention
这是 全序列计算，复杂度 O(L²)

在推理阶段（GPT 自回归生成）：
已生成序列 [x1, x2, ..., xt-1]
下一步只生成 xt
如果每次都重新计算 QKVT 对应整个历史序列，会浪费大量计算

✅ 解决方案 → KV Cache（Key/Value 缓存机制）
"""
class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x, seq_lengths=None, cached_k=None, cached_v=None):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cached_k is None:
            cached_k = k
            cached_v = v
        else:
            cached_k = torch.cat([cached_k, k], dim=2)
            cached_v = torch.cat([cached_v, v], dim=2)

        scores = torch.matmul(q, cached_k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        total_len = cached_k.size(2)
        causal_mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]
        if seq_lengths is not None:
            # padding mask
            padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
            padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
            final_mask = causal_mask | padding_mask
        else:
            final_mask = causal_mask

        scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, cached_v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att, cached_k, cached_v


"""
Attention 不是慢在算力而是慢在：内存访问和数值稳定处理
Softmax 是 attention 中最容易拖慢 GPU 的步骤

softmax 有三个步骤：(1) 求 man <主要作用是稳定数值，防止指数爆炸> (2) exp (3) sum + normalize
在 FP16 中 (1) exp对精度及其敏感 (2) sum是大规模reduction (3) mask会引入-inf
这会导致 梯度为NaN attention全0 loss突然爆炸
所以工程中 softmax 的关键步骤都是使用 FP32 累加的

FlashAttention 的核心是不把中间结果写回显存
传统的 attention 会显示的生成 L * L 的 score / softmax 矩阵,计算本身不满,但是显存读写,kernel启动和中间的 Tensor 成为瓶颈。
FlashAttention 通过按照 按 block 计算 QKᵀ → softmax → *V ,并在寄存器 / shared memory 中完成全部流程,把多个 kernel 融合成一个,从而极大减少显存带宽消耗和 kernel launch 次数。
Mask(causal / padding)在计算中直接融合,而不是事后用 -inf 修补。性能提升的本质来源是 内存访问模式优化 + kernel fusion,而不是数学公式变化。
"""

"""
TODO(P0,必须):
- 将基于 torch.cat 的 KV cache 改为预分配缓存
- cached_k / cached_v 形状统一为 [B, num_heads, max_seq_len, head_dim]
- 引入 cache_pos 指针,在 cache_pos 位置原地写入 k / v
- 推理模式下强制 seq_len == 1
- 使用 cache_pos 对历史 k / v 做切片,避免 O(T) 的 concat 操作
- 实现真正的 O(1) 增量解码推理
"""
class IncrementalKVAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _softmax(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def forward(self, x, seq_lengths=None, cached=None, cached_pos=None, is_training=False):

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if is_training is False:
            assert seq_len == 1

            if cached is None:
                cached = {
                    "k": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                    "v": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                }
                cached_pos = 0

            cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
            cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
            k = cached["k"][:, :, :cached_pos+1, :]
            v = cached["v"][:, :, :cached_pos+1, :]

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
        else:
            total_len = k.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device), diagonal=1)
            causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]

            if seq_lengths is not None:
                # padding mask
                padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
                padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
                final_mask = causal_mask | padding_mask
            else:
                final_mask = causal_mask

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = self._softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att, cached, cached_pos+1


"""
TODO(P1,性能关键):
- 合并 Q / K / V 为一个线性层:Linear(hidden_size, 3 * hidden_size)
- 使用 chunk(3, dim=-1) 拆分 fused QKV
- 删除自定义 _softmax 实现
- 使用 torch.softmax 或 scaled_dot_product_attention
"""
class FusedQKVAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len
        
        self.qkv_proj = nn.Linear(self.hidden_size, 3* self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x, seq_lengths=None, cached=None, cached_pos=None, is_training=False):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if is_training is False:
            assert seq_len == 1

            if cached is None:
                cached = {
                    "k": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                    "v": torch.zeros(batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=x.device, dtype=x.dtype),
                }
                cached_pos = 0

            cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
            cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
            k = cached["k"][:, :, :cached_pos+1, :]
            v = cached["v"][:, :, :cached_pos+1, :]

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
        else:
            total_len = k.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device), diagonal=1)
            causal_mask = causal_mask[None, None, :, :]  # [1,1,L,L]

            if seq_lengths is not None:
                # padding mask
                padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
                padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
                final_mask = causal_mask | padding_mask
            else:
                final_mask = causal_mask

            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            scores = scores.masked_fill(final_mask == 1, float("-inf"))

        att = torch.softmax(scores, dim=-1)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).contiguous()
        att = att.view(batch_size, seq_len, self.hidden_size)
        att = self.o_proj(att)

        return att, cached, cached_pos+1


"""
TODO(P2,FlashAttention / AMP 准备):
- 推理模式下不再构造 causal mask(单 token 天然满足因果性)
- padding mask 仅在训练模式下使用
- 确保 attention 相关张量布局为 [B, H, L, D] 且 contiguous
- 减少不必要的 transpose / permute 操作
"""
class FlashAttentionFusedAttention(nn.Module):
    """Fused QKV Attention，支持训练模式和 KV cache 增量解码。"""
    
    def __init__(self, config: DIYConfig):
        super().__init__()

        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout
        
        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_key_value_heads == 0

        self.head_dim = self.hidden_size // self.num_heads
        self.n_rep = self.num_heads // self.num_key_value_heads

        # fused QKV
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, 2 * self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        cached: Optional[Dict[str, torch.Tensor]] = None,
        cached_pos: Optional[int] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[int]]:
        """
        前向传播，支持训练和推理两种模式。
        
        Args:
            x: Input tensor，shape (batch_size, seq_len, hidden_size)
            seq_lengths: 每个样本的有效长度，shape (batch_size,)，用于 padding mask
            cached: KV cache dict，包含 "k" 和 "v"，用于增量解码
            cached_pos: Cache 位置，表示当前缓存到第几个位置
        
        Returns:
            (att_output, cached, cached_pos)
            - att_output: Attention 输出，shape (batch_size, seq_len, hidden_size)
            - cached: 更新后的 KV cache（推理时），训练时为 None
            - cached_pos: 更新后的 cache 位置（推理时），训练时为 None
        """
        batch_size, seq_len, _ = x.shape

        # fused QKV
        q = self.q_proj(x)
        kv = self.kv_proj(x)

        k, v = kv.chunk(2, dim=-1)

        # reshape为 [B,H,L,D] 并 contiguous
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        
        if freqs_cos is not None and freqs_sin is not None:
            from .rope import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        if self.training:
            # 训练模式：构造 causal + padding mask
            k = repeat_kv(k.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            v = repeat_kv(v.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            total_len = k.size(2)
            attn_mask = torch.triu(torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = attn_mask[None, None, :, :]  # [1,1,L,L]

            if seq_lengths is not None:
                padding_mask = torch.arange(total_len, device=x.device)[None, :] >= seq_lengths[:, None]  # [B,L]
                padding_mask = padding_mask[:, None, None, :]  # [B,1,1,L]
                attn_mask = attn_mask | padding_mask
        else:
            if cached is None:
                cached = {
                    "k": torch.zeros(batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim,
                                     device=x.device, dtype=x.dtype),
                    "v": torch.zeros(batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim,
                                     device=x.device, dtype=x.dtype),
                }
                if seq_len > 1:
                    cached["k"][:, :, :seq_len, :] = k
                    cached["v"][:, :, :seq_len, :] = v
                    cached_pos = seq_len
                else:
                    # 第一次 forward，seq_len=1，从位置 0 开始
                    cached_pos = 0
                    cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
                    cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
                    cached_pos = cached_pos + 1
                    k = cached["k"][:, :, :cached_pos, :]
                    v = cached["v"][:, :, :cached_pos, :]
            else:
                assert seq_len == 1
                cached["k"][:, :, cached_pos, :] = k[:, :, 0, :]
                cached["v"][:, :, cached_pos, :] = v[:, :, 0, :]
                cached_pos = cached_pos + 1
                k = cached["k"][:, :, :cached_pos + 1, :]
                v = cached["v"][:, :, :cached_pos + 1, :]

            k = repeat_kv(k.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            v = repeat_kv(v.transpose(1, 2), self.n_rep).transpose(1, 2)  # [B, num_heads, L, D]
            attn_mask = None

        # scaled dot product attention
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=self.dropout if self.training else 0.0)

        # 输出 reshape [B,H,L,D] -> [B,L,H*D]
        att = att.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        att = self.o_proj(att)
        att = self.resid_dropout(att)

        return att, cached, cached_pos if not self.training else None


if __name__ == "__main__":
    pass
