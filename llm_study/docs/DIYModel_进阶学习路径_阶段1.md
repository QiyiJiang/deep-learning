# 阶段一详细学习计划：旋转位置编码（RoPE）

**目标**：理解为什么需要位置编码，掌握 RoPE 的原理和实现，能在 DIYModel 中集成 RoPE。  
**前提**：已完成 DIYModel 基础学习，理解 Attention 的工作原理。  
**预计时间**：2-3 天

**建议**：新建一个单独的练习脚本（例如 `step_rope.py`），不要直接改 `DIYModel.py`，方便对比和回滚。

---

## 1.1 为什么需要位置编码

**目的**：理解 Transformer 的「置换不变性」问题，以及位置编码的作用。

**建议顺序**：

1. **理解问题**  
   - Transformer 的 self-attention 是**置换不变**的：打乱输入顺序，输出不变（除了位置信息）。
   - 但语言是有顺序的："猫吃鱼" ≠ "鱼吃猫"。
   - **问题**：没有位置信息，模型无法区分顺序。

2. **小实验验证**  
   - 用你的 DIYModel（当前没有位置编码），构造两个输入：
     - `input1 = torch.tensor([[1, 2, 3, 4, 5]])`
     - `input2 = torch.tensor([[5, 4, 3, 2, 1]])`（顺序相反）
   - 分别 forward，观察 `hidden_states` 是否相同（可能相同，因为只有 embedding，没有位置信息）。
   - **结论**：需要位置编码来区分顺序。

3. **位置编码的作用**  
   - 给每个位置一个**唯一标识**，让模型知道 token 的顺序。
   - 方式：可以是 learnable embedding、固定公式、或 RoPE（相对位置）。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 导入必要的模块
import torch
from llm_study import DIYModel, DIYConfig

# 创建配置和模型
config = DIYConfig(
    vocab_size=6400,
    hidden_size=256,
    num_layers=2,
    num_heads=4,
    max_seq_len=128,
    dropout=0.1
)
model = DIYModel(config)
model.eval()

# 构造两个顺序相反的输入
input1 = torch.tensor([[1, 2, 3, 4, 5]])
input2 = torch.tensor([[5, 4, 3, 2, 1]])

# 分别 forward
with torch.no_grad():
    hidden1 = model(input1)
    hidden2 = model(input2)

# 检查输出是否相同（可能相同，因为没有位置编码）
print("Hidden1 shape:", hidden1.shape)
print("Hidden2 shape:", hidden2.shape)
print("Are they equal?", torch.allclose(hidden1, hidden2))
```

**自检**：
- 理解为什么 Transformer 需要位置编码（置换不变性问题）。
- 能解释「置换不变性」的含义（打乱顺序，输出不变）。
- 理解位置编码的作用（给每个位置唯一标识）。

**学习时间**：约 30 分钟

---

## 1.2 绝对位置编码 vs 相对位置编码

**目的**：理解两种位置编码方式的区别，以及为什么 RoPE（相对位置）更适合长序列。

**建议顺序**：

1. **理解绝对位置编码**  
   - 方式：每个位置有一个独立的 embedding，`pos_emb[0]`, `pos_emb[1]`, ...
   - 实现：`pos_embedding = nn.Embedding(max_seq_len, hidden_size)`
   - 应用：`hidden = token_emb + pos_embedding(position_ids)`
   - **缺点**：训练时只见过固定长度（如 512），推理时无法外推到更长序列（没见过的位置没有 embedding）。

2. **理解相对位置编码**  
   - 方式：编码的是「位置之间的相对关系」，而不是绝对位置。
   - 例如：位置 5 和位置 3 的相对距离是 2，位置 1005 和位置 1003 的相对距离也是 2。
   - **优点**：可以外推到更长序列（相对关系不变）。

3. **实现简单的绝对位置编码（可选）**  
   - 在 `DIYModel` 中添加绝对位置编码：
     ```python
     self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
     ```
   - 在 forward 中应用：
     ```python
     position_ids = torch.arange(seq_len, device=input_ids.device)
     pos_emb = self.pos_embedding(position_ids)
     hidden = token_emb + pos_emb
     ```
   - **观察**：训练时用短序列（如 128），推理时用长序列（如 256），效果可能变差。

4. **理解 RoPE 的优势**  
   - RoPE 是相对位置编码的一种实现方式。
   - **优势**：
     - 可以外推到更长序列（相对位置关系不变）
     - 不需要额外的 embedding 层（用旋转矩阵实现）
     - 在 attention 计算中直接应用，更高效

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 实现简单的绝对位置编码
class DIYModelWithAbsPos(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        # ... 其他层
    
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        token_emb = self.embed_tokens(input_ids)
        
        # 应用绝对位置编码
        position_ids = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.pos_embedding(position_ids)
        hidden = token_emb + pos_emb
        
        # ... 后续处理
        return hidden
```

**自检**：
- 理解绝对位置编码和相对位置编码的区别。
- 知道 RoPE 的优势（可以外推、不需要额外 embedding）。
- 能解释为什么相对位置编码更适合长序列。

**学习时间**：约 1 小时

---

## 1.3 旋转位置编码（RoPE）原理与实现

**目的**：理解 RoPE 的数学原理，并能在 DIYModel 中实现。

**建议顺序**：

1. **理解数学原理**  
   - **频率计算**：$\theta_i = 1 / \text{base}^{2i/d}$，其中 i 是维度索引（0 到 d/2-1），d 是 head_dim。
   - **旋转角度**：位置 m 的向量，旋转角度是 $m \theta_i$。
   - **旋转矩阵**：用 cos/sin 实现旋转：
     ```
     [cos(mθ)  -sin(mθ)] [q0]
     [sin(mθ)   cos(mθ)] [q1]
     ```
   - **相对位置**：位置 m 和 n 的相对旋转角度是 $(m-n)\theta$，与绝对位置无关。

2. **实现 precompute_freqs_cis**  
   - **输入**：`dim`（head_dim）、`end`（最大位置）、`rope_base`（基础频率，如 1e6）。
   - **步骤**：
     1. 计算基础频率：`freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))`
     2. 计算所有位置的频率矩阵：`freqs = torch.outer(torch.arange(end), freqs)`
     3. 生成 cos/sin：`freqs_cos = torch.cos(freqs)`，`freqs_sin = torch.sin(freqs)`
     4. 重复一次以匹配完整维度：`freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)`
   - **返回**：`(freqs_cos, freqs_sin)`，shape 都是 `(max_seq_len, head_dim)`
   - **注意**：
     - `torch.arange(0, dim, 2)` 只取偶数索引（0, 2, 4, ...），因为每两个维度共享一个频率
     - `torch.outer` 计算外积，得到所有位置的频率矩阵

3. **实现 apply_rotary_pos_emb**  
   - **输入**：`q, k`（shape `(batch, num_heads, seq_len, head_dim)`）、`cos, sin`（shape `(seq_len, head_dim)`）。
   - **步骤**：
     1. 定义 `rotate_half`：把后一半维度取负并与前一半拼接：`[-x[d/2:], x[:d/2]]`
     2. 应用旋转：`q_rot = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)`
     3. 对 k 同样操作
   - **返回**：`(q_rot, k_rot)`
   - **注意**：
     - `cos` 和 `sin` 需要 unsqueeze 到正确的维度（通常是第 1 维，即 num_heads 维度）
     - `rotate_half` 的作用是把向量分成两半，后一半取负，前一半不变，然后拼接

4. **在 DIYModel 中集成 RoPE**  
   - **在 `DIYModel.__init__` 里**：
     - 预计算 `freqs_cos, freqs_sin`（用 `precompute_freqs_cis`）
     - 用 `register_buffer` 注册为 buffer（不参与训练，但会随模型保存）
     - 需要知道 `head_dim`：`head_dim = config.hidden_size // config.num_heads`
   - **在 Attention 里**：
     - 计算 Q/K 后，应用 RoPE：`q, k = apply_rotary_pos_emb(q, k, cos, sin)`
     - 然后再做 attention
     - **注意**：需要从 `DIYModel` 传递 `freqs_cos` 和 `freqs_sin` 到 `Attention`

5. **验证 RoPE 集成**  
   - **测试前向传播**：
     - 构造一个输入，测试模型是否能正常 forward
     - 检查输出 shape 是否正确
   - **测试外推能力**：
     - 训练时用短序列（如 128），推理时用长序列（如 256）
     - 观察是否还能正常工作（不会报错，输出合理）
   - **对比实验**：
     - 有/无 RoPE 在长序列上的效果差异

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
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
    
    # unsqueeze 到 num_heads 维度（第 1 维）
    cos = cos.unsqueeze(1)  # shape: (seq_len, 1, head_dim)
    sin = sin.unsqueeze(1)  # shape: (seq_len, 1, head_dim)
    
    # 应用旋转
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot

# 在 DIYModel.__init__ 中添加
class DIYModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... 现有代码 ...
        
        # 预计算 RoPE
        head_dim = config.hidden_size // config.num_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim,
            end=config.max_seq_len,
            rope_base=1e6
        )
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

# 在 Attention 中应用 RoPE
class FlashAttentionFusedAttention(nn.Module):
    def forward(self, x, ..., freqs_cos=None, freqs_sin=None):
        # ... 计算 Q/K/V ...
        
        # 应用 RoPE
        if freqs_cos is not None and freqs_sin is not None:
            seq_len = x.shape[1]
            q, k = apply_rotary_pos_emb(
                q, k,
                freqs_cos[:seq_len],  # 只取当前序列长度的部分
                freqs_sin[:seq_len]
            )
        
        # ... 后续 attention 计算 ...
```

**自检**：
- 能实现 `precompute_freqs_cis` 函数，预计算所有位置的 cos/sin 值。
- 能实现 `apply_rotary_pos_emb` 函数，对 Q/K 应用旋转。
- 能在 `DIYModel` 中集成 RoPE，在 Attention 里正确应用。
- 能验证 RoPE 的外推能力（训练时用短序列，推理时用长序列）。

**常见问题**：
- **Q: `freqs_cos` 和 `freqs_sin` 的 shape 不对？**  
  A: 检查 `precompute_freqs_cis` 的返回值，确保是 `(max_seq_len, head_dim)`。
- **Q: `apply_rotary_pos_emb` 的维度对齐错误？**  
  A: 确保 `cos` 和 `sin` 正确 unsqueeze 到 num_heads 维度（第 1 维）。
- **Q: RoPE 没有效果？**  
  A: 确保 RoPE 在 attention 计算**之前**应用（先旋转 Q/K，再做 attention）。

**学习时间**：约 1-2 天（包括实现和调试）

---

## 阶段一汇总：完成 1.1 + 1.2 + 1.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 理解位置编码的必要性**

- 能解释为什么 Transformer 需要位置编码（置换不变性问题）。
- 理解绝对位置编码和相对位置编码的区别。
- 知道 RoPE 的优势（可以外推、不需要额外 embedding）。

**2. RoPE 实现**

- 有 `precompute_freqs_cis` 函数，能预计算所有位置的 cos/sin 值。
- 有 `apply_rotary_pos_emb` 函数，能对 Q/K 应用旋转。
- `DIYModel` 中集成了 RoPE，在 Attention 里正确应用。
- 用 `register_buffer` 注册 `freqs_cos` 和 `freqs_sin`（不参与训练，但会随模型保存）。

**3. 验证外推能力**

- 训练时用短序列（如 128），推理时用长序列（如 256），模型仍能正常工作。
- 理解为什么 RoPE 能外推（相对位置关系不变）。
- 对比实验：有/无 RoPE 在长序列上的效果差异。

**4. 代码质量**

- 代码结构清晰，有适当的注释。
- 能正确处理维度对齐（cos/sin 的 unsqueeze）。
- 能正确处理不同序列长度（用 `freqs_cos[:seq_len]` 切片）。

**5. 下一步**

- 有了 RoPE，模型就有了位置信息，可以进入 **阶段二**：调整模型规模，让它能真正学到语义信息。

---

## 学习建议

1. **先理解原理，再动手实现**：先看数学原理，理解「为什么」要用旋转矩阵，再写代码。
2. **小步快跑**：先实现 `precompute_freqs_cis`，验证输出 shape 正确；再实现 `apply_rotary_pos_emb`，验证维度对齐；最后集成到模型中。
3. **多做实验**：对比有/无 RoPE 的效果，理解位置编码的作用。
4. **记录笔记**：记录遇到的问题、解决方案、以及 RoPE 的效果。

若某一步卡住，优先检查：
- `freqs_cos` 和 `freqs_sin` 的 shape 是否正确（`(max_seq_len, head_dim)`）；
- `apply_rotary_pos_emb` 的维度对齐是否正确（cos/sin 需要 unsqueeze 到正确的维度）；
- RoPE 是否在 attention 计算**之前**应用（先旋转 Q/K，再做 attention）；
- `head_dim` 的计算是否正确（`hidden_size // num_heads`）。
