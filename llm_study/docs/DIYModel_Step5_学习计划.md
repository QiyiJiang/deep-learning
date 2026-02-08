# 第五步详细学习计划：完善接口与细节（按需选做）

**目标**：让模型更完整、更易用、更易维护，包括支持 attention_mask、可配置的 RMSNorm eps、权重共享、以及类型注解和文档。  
**前提**：已完成第四步，有可用的 `DIYForCausalLM` 和 `DIYConfig`，且能正常保存/加载。  
**说明**：这一步的各个子项是**可选的**，按需选做即可，不要求全部完成。

---

## 5.1 支持 attention_mask

**目的**：让接口更符合常见用法：接收 `attention_mask`（0/1 mask），在内部转换成 `seq_lengths` 传给底层，底层仍用现有的 `seq_lengths` 做 padding mask。

**建议顺序**：

1. **理解 attention_mask 与 seq_lengths 的关系**  
   - `attention_mask`：shape `(batch_size, seq_len)`，1 表示有效 token，0 表示 padding。
   - `seq_lengths`：shape `(batch_size,)`，每个样本的有效长度（整数）。
   - 转换：`seq_lengths = attention_mask.sum(dim=1)`，即每行有多少个 1。

2. **在 DIYForCausalLM.forward 里接收 attention_mask**  
   - 参数：`forward(self, input_ids, attention_mask=None, seq_lengths=None, past_key_values=None, use_cache=False)`。
   - 逻辑：如果传了 `attention_mask`，从中得到 `seq_lengths`；如果传了 `seq_lengths`，直接用；如果都没传，`seq_lengths=None`（表示没有 padding）。
   - 例如：
     ```python
     if attention_mask is not None:
         seq_lengths = attention_mask.sum(dim=1)
     # 然后传给 self.model(..., seq_lengths=seq_lengths, ...)
     ```

3. **保持底层接口不变**  
   - `DIYModel` 和 `ModelBlock` 的接口不变，仍用 `seq_lengths`。
   - 只在 `DIYForCausalLM` 这一层做转换，这样底层代码不需要改。

4. **自检**  
   - 用 `attention_mask` 调用：`logits = model(input_ids, attention_mask=mask)`，能正常运行。
   - 用 `seq_lengths` 调用：`logits = model(input_ids, seq_lengths=lengths)`，行为一致。
   - 可选：对比两种方式在相同输入下的结果是否一致。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
def forward(self, input_ids, attention_mask=None, seq_lengths=None, past_key_values=None, use_cache=False):
    # 从 attention_mask 得到 seq_lengths
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1)
    
    out = self.model(input_ids, seq_lengths=seq_lengths, past_key_values=past_key_values, 
                     use_cache=use_cache)
    # ... 后续逻辑
```

---

## 5.2 RMSNorm 的 eps 可配置

**目的**：把 RMSNorm 的 `eps` 参数放到 Config 里，便于调整和复现。

**建议顺序**：

1. **在 DIYConfig 里加 rms_norm_eps**  
   - 例如：`rms_norm_eps: float = 1e-5`（默认值和 RMSNorm 的默认值一致）。

2. **修改 DIYModel.__init__**  
   - 构造 `RMSNorm` 时传入 `eps=config.rms_norm_eps`。
   - 例如：`self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)`。
   - 注意：`ModelBlock` 里也有 `RMSNorm`（`attention_norm` 和 `feedforward_norm`），也需要传 `eps`。可以在 `ModelBlock.__init__` 里也接收一个 `rms_norm_eps` 参数，或直接从 config 传。

3. **可选：理解 eps 的作用**  
   - `eps` 在分母上：`x / sqrt(mean(x²) + eps)`，防止除零和数值不稳定。
   - 通常 `1e-5` 或 `1e-6` 即可，太小可能数值不稳定，太大可能影响归一化效果。

4. **自检**  
   - 用不同 `rms_norm_eps` 的 config 构造模型，能正常运行。
   - 可选：对比不同 `eps` 值对训练 loss 的影响（通常影响很小）。

**可参考的代码片段（只作提示）**：

```python
# 在 DIYConfig 里
@dataclass
class DIYConfig:
    # ... 其他参数
    rms_norm_eps: float = 1e-5

# 在 DIYModel.__init__ 里
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# 在 ModelBlock.__init__ 里（如果需要）
self.attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
self.feedforward_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
```

---

## 5.3 权重共享（embed 与 lm_head）

**目的**：让 `embed_tokens` 和 `lm_head` 共享同一块权重，减少参数量，且可能有助于训练稳定。

**建议顺序**：

1. **理解权重共享**  
   - 通常：`embed_tokens.weight` shape `(vocab_size, hidden_size)`，`lm_head.weight` shape `(vocab_size, hidden_size)`。
   - 共享：`lm_head.weight = embed_tokens.weight`，让两个层用同一块参数。
   - 效果：参数量减少（从 `2 * vocab_size * hidden_size` 变成 `vocab_size * hidden_size`），且输入和输出的 embedding 空间对齐。

2. **在 DIYForCausalLM.__init__ 里绑定权重**  
   - 先构造 `self.model` 和 `self.lm_head`。
   - 然后：`self.lm_head.weight = self.model.embed_tokens.weight`。
   - 注意：`lm_head` 的构造要在绑定之前，且 `lm_head` 的 `bias=False`（如果之前不是，需要改成 `False`，因为共享权重时通常不用 bias）。

3. **可选：观察参数量变化**  
   - 绑定前：`sum(p.numel() for p in model.parameters())`。
   - 绑定后：再算一次，应减少 `vocab_size * hidden_size` 个参数。
   - 打印：`print(f"Total parameters: {total_params / 1e6:.2f}M")`。

4. **可选：对比训练曲线**  
   - 用相同数据训练「共享权重」和「不共享权重」两个版本，观察 loss 下降速度是否不同（通常共享权重可能略慢，但参数量更少）。

5. **自检**  
   - 权重绑定后，`lm_head.weight is embed_tokens.weight` 应为 `True`（是同一个对象）。
   - 模型能正常 forward 和 backward。
   - 参数量确实减少了。

**可参考的代码片段（只作提示）**：

```python
def __init__(self, config: DIYConfig):
    super().__init__()
    self.vocab_size = config.vocab_size
    self.model = DIYModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    # 权重共享
    self.lm_head.weight = self.model.embed_tokens.weight
```

---

## 5.4 类型注解与 docstring

**目的**：让接口一目了然，便于「几个月后的自己」和他人快速看懂参数、返回值、以及何时返回什么。

**建议顺序**：

1. **给 forward 参数加类型注解**  
   - 导入：`from typing import Optional, Tuple, List`。
   - 参数类型：
     - `input_ids: torch.Tensor`（或更精确：`input_ids: torch.LongTensor`）。
     - `seq_lengths: Optional[torch.Tensor] = None`。
     - `past_key_values: Optional[List[Tuple[dict, int]]] = None`（根据你的 cache 格式）。
     - `use_cache: bool = False`。
   - 返回值类型：
     - 训练时：`torch.Tensor`（logits）。
     - 推理且 use_cache 时：`Tuple[torch.Tensor, List[Tuple[dict, int]]]`（logits 和 presents）。

2. **写简短的 docstring**  
   - 格式：用三引号 `"""..."""`，第一行简短说明，然后空一行，再写参数和返回值说明。
   - 例如：
     ```python
     def forward(self, input_ids: torch.Tensor, ...):
         """
         前向传播，返回 logits（训练）或 (logits, presents)（推理且 use_cache）。
         
         Args:
             input_ids: Token IDs，shape (batch_size, seq_len)
             seq_lengths: 每个样本的有效长度，shape (batch_size,)，None 表示无 padding
             past_key_values: KV cache，每层一个 (cached_dict, cached_pos) tuple
             use_cache: 是否返回 presents（用于增量解码）
         
         Returns:
             训练时返回 logits，shape (batch_size, seq_len, vocab_size)
             推理且 use_cache=True 时返回 (logits, presents)
         """
     ```

3. **给类也加 docstring**  
   - 在 `DIYModel` 和 `DIYForCausalLM` 的类定义下加一行说明：这个类做什么、主要组件是什么。

4. **自检**  
   - 类型注解和 docstring 能让「只看接口不看实现」的人理解用法。
   - 可选：用 IDE 的自动补全和类型检查，确认类型注解正确。

**可参考的代码片段（只作提示）**：

```python
from typing import Optional, Tuple, List, Union

class DIYForCausalLM(nn.Module):
    """DIY 因果语言模型：DIYModel + lm_head，用于训练和生成。"""
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[dict, int]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[dict, int]]]]:
        """
        前向传播。
        
        Args:
            input_ids: Token IDs，shape (batch_size, seq_len)
            attention_mask: Attention mask，1=有效，0=padding，shape (batch_size, seq_len)
            seq_lengths: 每个样本的有效长度，shape (batch_size,)
            past_key_values: KV cache，用于增量解码
            use_cache: 是否返回 presents
        
        Returns:
            训练时返回 logits (batch_size, seq_len, vocab_size)
            推理且 use_cache=True 时返回 (logits, presents)
        """
        # ... 实现
```

---

## 建议时间与自测清单

- **5.1**：约 15～20 分钟。自测：能用 `attention_mask` 调用，行为与 `seq_lengths` 一致。
- **5.2**：约 10～15 分钟。自测：Config 里有 `rms_norm_eps`，构造模型时传入 RMSNorm。
- **5.3**：约 15～20 分钟。自测：权重绑定后参数量减少，`lm_head.weight is embed_tokens.weight` 为 True。
- **5.4**：约 20～30 分钟。自测：类型注解和 docstring 能让接口一目了然。

---

## 第五步汇总：完成 5.1 + 5.2 + 5.3 + 5.4 后的结果

做完上面各小节后（按需选做），你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. attention_mask 支持**

- `DIYForCausalLM.forward` 能接收 `attention_mask`，内部转换成 `seq_lengths` 传给底层。
- 底层接口不变，仍用 `seq_lengths`，只在封装层做转换。
- 用 `attention_mask` 和 `seq_lengths` 调用行为一致。

**2. RMSNorm eps 可配置**

- `DIYConfig` 里有 `rms_norm_eps` 参数。
- 构造 `DIYModel` 和 `ModelBlock` 里的 `RMSNorm` 时传入 `eps=config.rms_norm_eps`。
- 便于调整和复现不同配置。

**3. 权重共享**

- `lm_head.weight = embed_tokens.weight`，两个层共享同一块参数。
- 参数量减少 `vocab_size * hidden_size`。
- 模型能正常训练和推理。

**4. 类型注解与 docstring**

- `forward` 方法的参数和返回值都有类型注解。
- 类和主要方法都有简短的 docstring，说明用途、参数、返回值。
- 接口一目了然，便于后续扩展和维护。

**5. 完成整个学习路线**

- 完成第一步到第五步后，你就有了一个**完整的、可训练、可生成、可保存、可复现、易维护**的语言模型。
- 可以在此基础上继续扩展：更大的模型、更多层、MoE、LoRA 等。

若某一步卡住，优先检查：  
- attention_mask 转 seq_lengths 的逻辑是否正确（`attention_mask.sum(dim=1)`）；  
- RMSNorm 的 eps 是否在所有用到的地方都传入了（DIYModel 的 norm、ModelBlock 的 attention_norm 和 feedforward_norm）；  
- 权重共享后是否还能正常 backward（共享权重时梯度会累加，这是正常的）。
