import math
from sympy import tensor
import torch
from torch import nn

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

if __name__ == "__main__":
    pass