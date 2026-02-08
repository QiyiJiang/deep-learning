# 第三步详细学习计划：自回归生成与 KV cache 的正确用法

**目标**：会用现有的 `past_key_values` / `presents` 做增量解码（逐 token 生成），并统一用 `model.train()` / `model.eval()` 控制训练/推理分支，不再手动传 `is_training`。  
**前提**：已完成第二步，有可用的 `DIYForCausalLM`，且能正常训练。  
**建议**：新建一个单独脚本（例如 `step3_generation.py`），先做无 cache 版本，再做有 cache 版本，最后改 `DIYModel` 用 `self.training`。

---

## 3.1 先实现「无 cache」的逐 token 生成

**目的**：理解自回归生成的本质：每次用「已生成序列 + 新 token」预测下一个，且每步都重算整段 attention，复杂度随步数平方增长。

**建议顺序**：

1. **准备 prompt**  
   - 一段初始序列（如 5～10 个 token），作为生成的起点。例如 `prompt = torch.tensor([[1, 42, 7, 0, 15]], dtype=torch.long)`，shape 为 `(1, 5)`（batch_size=1，方便简单）。
   - 用 `model.eval()` 切换到推理模式（dropout 关闭、batch norm 用统计值等）。

2. **写生成循环（无 cache）**  
   - 初始化：`generated = prompt.clone()`（或 `generated = prompt`，看是否需要独立副本）。
   - 循环（例如生成 10 个新 token）：
     - **前向**：`logits = model(generated)`（注意：每次都是整段 `generated` 送进去）。
     - **取最后一个位置的 logits**：`next_logits = logits[:, -1, :]`，shape 为 `(1, vocab_size)`。
     - **选下一个 token**：
       - **贪心**：`next_token = next_logits.argmax(dim=-1)`，shape 为 `(1,)`。
       - **采样**（可选）：`next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)`。
     - **拼到序列后**：`generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)`，或 `generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)`（注意维度对齐）。
   - 循环结束后，`generated` 就是 prompt + 生成的 token。

3. **理解复杂度**  
   - 每步都要把整段 `generated` 送进 model，attention 计算量是 `O(seq_len²)`。
   - 生成 N 个 token 后，总计算量约为 `O(N²)`（或更精确：`O(1² + 2² + ... + N²) = O(N³)`）。
   - 这就是为什么需要 KV cache：把历史 K/V 存下来，新 token 只和「历史 + 当前」做 attention，每步只算 `O(1)`。

4. **自检**  
   - 生成循环能跑通，`generated.shape` 从 `(1, prompt_len)` 逐步增长到 `(1, prompt_len + num_generated)`。
   - 打印每步生成的 token id，确认序列在增长。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
model.eval()
prompt = torch.tensor([[1, 42, 7, 0, 15]], dtype=torch.long)  # (1, 5)
generated = prompt.clone()
num_generate = 10

for step in range(num_generate):
    logits = model(generated)  # 每次都是整段
    next_logits = logits[:, -1, :]  # (1, V)
    next_token = next_logits.argmax(dim=-1)  # (1,)
    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    print(f"step {step}, generated token: {next_token.item()}")
```

---

## 3.2 再用 use_cache 做增量解码

**目的**：用 KV cache 加速生成：第一次 forward 整段 prompt 并拿到 `presents`，之后每步只送 1 个新 token，把上一轮的 `presents` 当 `past_key_values` 传入，每步只算 `O(1)` 的 attention。

**建议顺序**：

1. **理解你当前的 cache 格式**  
   - `DIYModel` 的 `past_key_values`：每层一个 tuple `(cached, cached_pos)`，其中：
     - `cached` 是一个 dict：`{"k": tensor, "v": tensor}`（shape 见 Attention 代码）。
     - `cached_pos` 是整数，表示当前缓存到第几个位置。
   - `presents`：和 `past_key_values` 格式一样，是 `List[Tuple[cached_dict, cached_pos]]`。

2. **第一次 forward（整段 prompt）**  
   - `model.eval()`，`use_cache=True`。
   - `logits, presents = model(prompt, use_cache=True)`（注意：你的 `DIYForCausalLM.forward` 在推理且 use_cache 时应返回 `(logits, presents)`）。
   - 取最后一个位置的 logits，选第一个新 token：`next_token = logits[:, -1, :].argmax(dim=-1)`。
   - 初始化 `generated = prompt.clone()`，并拼上第一个 token：`generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)`。

3. **后续每步（增量）**  
   - 只送「新生成的 1 个 token」：`new_token = next_token.unsqueeze(0)`，shape 为 `(1, 1)`。
   - 把上一轮的 `presents` 当作 `past_key_values` 传入：`logits, presents = model(new_token, past_key_values=presents, use_cache=True)`。
   - 取最后一个位置的 logits（实际只有 1 个位置），选下一个 token，拼到 `generated`。
   - 更新 `presents` 为这一轮的返回值，用于下一步。

4. **可选：对比无 cache 和有 cache 的结果**  
   - 用相同的 prompt 和 seed，分别跑无 cache 和有 cache 版本，对比前几步的 logits 或生成的 token 是否一致（应一致，cache 只是加速，不改变结果）。

5. **自检**  
   - 有 cache 版本能跑通，生成的序列长度正确。
   - 可选：对比无 cache 和有 cache 在相同 prompt 下的前几步 logits，应一致（或非常接近，允许浮点误差）。

**可参考的代码片段（只作提示）**：

```python
model.eval()
prompt = torch.tensor([[1, 42, 7, 0, 15]], dtype=torch.long)  # (1, 5)
generated = prompt.clone()
num_generate = 10

# 第一次：整段 prompt
logits, presents = model(prompt, use_cache=True)
next_token = logits[:, -1, :].argmax(dim=-1)
generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

# 后续每步：只送 1 个 token
for step in range(num_generate - 1):
    new_token = next_token.unsqueeze(0).unsqueeze(1)  # (1, 1)
    logits, presents = model(new_token, past_key_values=presents, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    print(f"step {step + 1}, generated token: {next_token.item()}")
```

注意：你的 `DIYForCausalLM.forward` 需要把 `past_key_values` 和 `use_cache` 传给 `self.model(...)`，且推理时返回 `(logits, presents)`。若当前接口不完整，先补上这些参数传递。

---

## 3.3 用 self.training 替代 is_training 参数

**目的**：统一用 PyTorch 的 `model.train()` / `model.eval()` 控制训练/推理分支，不再手动传 `is_training`，让接口更符合 PyTorch 惯例。

**建议顺序**：

1. **理解 self.training**  
   - `nn.Module` 有一个属性 `self.training`，调用 `model.train()` 时设为 `True`，`model.eval()` 时设为 `False`。
   - 训练时通常先 `model.train()`，推理时先 `model.eval()`。

2. **改 DIYModel.forward**  
   - 把 `is_training` 参数改为可选（或删除），在函数开头用 `is_training = self.training`（若保留参数为可选 override，则 `is_training = self.training if is_training is None else is_training`）。
   - 这样调用方只需 `model.train()` / `model.eval()`，不再传 `is_training`。

3. **改调用方**  
   - 训练循环：在循环前 `model.train()`，调用 `model(input_ids)` 时不再传 `is_training=True`。
   - 生成循环：在循环前 `model.eval()`，调用 `model(...)` 时不再传 `is_training=False`。
   - 若 `DIYForCausalLM.forward` 也接收 `is_training`，同样改为用 `self.training`（或删除该参数，直接传给 `self.model` 时也不传）。

4. **自检**  
   - 训练循环：`model.train()` 后，`model.training` 应为 `True`，且训练行为正常（dropout 开启等）。
   - 生成循环：`model.eval()` 后，`model.training` 应为 `False`，且推理行为正常（dropout 关闭、cache 可用等）。
   - 可选：对比改前改后，在相同 seed 下训练 loss 和生成结果应一致。

**可参考的代码片段（只作提示）**：

```python
# 在 DIYModel.forward 里
def forward(self, input_ids, seq_lengths=None, past_key_values=None, use_cache=False, is_training=None):
    # 用 self.training，但允许 override（可选）
    if is_training is None:
        is_training = self.training
    # 或直接：is_training = self.training
    # ... 后续逻辑不变

# 调用方
model.train()  # 训练前
logits = model(input_ids)  # 不再传 is_training=True

model.eval()  # 推理前
logits, presents = model(prompt, use_cache=True)  # 不再传 is_training=False
```

---

## 建议时间与自测清单

- **3.1**：约 20～30 分钟。自测：无 cache 生成循环能跑通，序列逐步增长，理解每步重算整段 attention 的复杂度。
- **3.2**：约 30～40 分钟。自测：有 cache 版本能跑通，生成的序列长度正确；可选：与无 cache 版本在相同 prompt 下前几步 logits 一致。
- **3.3**：约 15～20 分钟。自测：`model.train()` / `model.eval()` 后 `model.training` 正确，训练和生成行为正常。

---

## 第三步汇总：完成 3.1 + 3.2 + 3.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 无 cache 生成**

- 能写一个循环：每次用整段已生成序列 forward，取最后一个位置的 logits，选下一个 token，拼到序列后。
- 理解这样每步都重算整段 attention，复杂度随步数平方（或立方）增长。

**2. 有 cache 增量解码**

- 第一次 forward 整段 prompt，拿到 `logits, presents`。
- 后续每步只送 1 个新 token，把上一轮的 `presents` 当 `past_key_values` 传入，得到新的 `logits, presents`。
- 理解这样每步只算 `O(1)` 的 attention，总复杂度从 `O(N²)` 降到 `O(N)`。
- 理解你当前 `past_key_values` 的格式：`List[Tuple[cached_dict, cached_pos]]`，每层一个 tuple。

**3. 统一用 self.training**

- `DIYModel.forward` 不再需要（或可选）`is_training` 参数，内部用 `self.training`。
- 调用方只需 `model.train()` / `model.eval()`，不再手动传 `is_training`。
- 训练和生成都通过 `model.training` 自动走对应分支。

**4. 下一步**

- 有了「能生成」的验证和统一的 train/eval 接口，就可以进入 **第四步**：引入 Config、保存/加载 state_dict、把 config 存进 checkpoint，便于复现和继续改。

若某一步卡住，优先检查：  
- 推理时 `seq_len` 是否为 1（有 cache 的增量步骤）；  
- `past_key_values` 的格式是否与 `presents` 一致（每层是 `(cached_dict, cached_pos)` tuple）；  
- `DIYForCausalLM.forward` 是否把 `past_key_values` 和 `use_cache` 正确传给 `self.model(...)`，且推理时返回 `(logits, presents)`。
