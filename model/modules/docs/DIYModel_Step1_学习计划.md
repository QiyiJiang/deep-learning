# 第一步详细学习计划：从 hidden_states 到 logits

目标：在现有 `DIYModel` 基础上，加上语言模型头（lm_head），得到 logits，能算 loss 并 backward；最后封装成「完整语言模型」类。  
**建议**：新建一个单独的练习脚本（例如 `step1_lm_head.py`），不要直接改 `DIYModel.py`，方便对比和回滚。

---

## 1.1 加 lm_head，得到 logits

**目的**：理解「最后一层」就是把每个位置的 hidden 映射到词表大小的分数。

**建议顺序**：

1. **在脚本里导入**  
   需要：`torch`、`nn`，以及你的 `DIYModel`（根据你项目里 DIYModel 所在路径写 import）。

2. **设一组和 DIYModel 一致的超参**（方便后面封装）  
   例如：
   - `vocab_size = 6400`（或你词表大小）
   - `hidden_size = 256`
   - `num_layers = 2`
   - `num_heads = 4`
   - `max_seq_len = 128`
   - `dropout = 0.1`

3. **实例化 DIYModel**  
   用上面参数构造 `model`，再构造一个 **单独的** `lm_head`：
   - 输入维度 = `hidden_size`
   - 输出维度 = `vocab_size`
   - 一般用 `bias=False`（和常见 LLM 一致）

   **形状对应**：  
   `hidden_states` 的最后一维是 `hidden_size`，lm_head 把它变成 `vocab_size`，所以：
   - `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`

4. **造一小份假数据**  
   - `batch_size=2, seq_len=10` 即可。
   - `input_ids = torch.randint(0, vocab_size, (2, 10))`。

5. **前向**  
   - 调用 `model(...)` 得到 `hidden_states`（训练模式即可，`is_training=True`）。
   - 再对 `hidden_states` 做一次 `lm_head(...)` 得到 `logits`。

6. **检查 shape**  
   - `hidden_states.shape` 应为 `(2, 10, hidden_size)`。
   - `logits.shape` 应为 `(2, 10, vocab_size)`。  
   每个 (batch_idx, seq_idx) 位置都是一个长度为 `vocab_size` 的向量，表示「该位置下一个 token 是词表里各 id 的分数」。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 前向取 hidden_states（这里假设你返回的是 hidden_states）
hidden_states = model(input_ids, is_training=True)
# 再过 lm_head
logits = lm_head(hidden_states)
# 检查
print(hidden_states.shape, logits.shape)
```

---

## 1.2 用 logits 算 loss 并 backward

**目的**：理解「next-token prediction」的 label 怎么取，以及 padding 怎么用 `ignore_index` 排除。

**建议顺序**：

1. **定 label 的取法**  
   - 常见做法：预测「下一个 token」，即用 `input_ids[:, 1:]` 作为 target，logits 用 `logits[:, :-1, :]`（即去掉最后一个位置，因为最后一个位置没有「下一个」）。
   - 另一种：同长度，即 `labels = input_ids`，logits 每个位置预测「当前 token」；此时 loss 里要对齐 shape：logits 是 `(B, L, V)`，labels 是 `(B, L)`，做 cross_entropy 时要把 logits 展成 `(B*L, V)`，labels 展成 `(B*L)`。

2. **选一种先实现**（建议先做「下一个 token」）  
   - `logits_slice = logits[:, :-1, :]`  # (B, L-1, V)  
   - `labels_slice = input_ids[:, 1:]`   # (B, L-1)  
   - 定义 `padding_id`（例如 0），padding 位置在 labels 里填 `padding_id`，算 loss 时用 `ignore_index=padding_id`。

3. **调用 cross_entropy**  
   - `F.cross_entropy` 的输入：  
     - 第一项：logits 要变成 `(N, vocab_size)`，即把 batch 和 seq 两维压成一维。  
     - 第二项：labels 变成 `(N,)`，且 dtype 为 long。  
   - 参数：`ignore_index=padding_id`（若你没有 padding，可先不设或设一个不会出现的 id）。

4. **backward**  
   - 先 `optimizer.zero_grad()`（若有 optimizer），再 `loss.backward()`。  
   - 若只验证反传，可以没有 optimizer，只做 `loss.backward()`，检查 `model` 和 `lm_head` 的参数是否有 `grad`。

**可参考的代码片段（只作提示）**：

```python
# 下一个 token 的 label
logits_slice = logits[:, :-1, :].reshape(-1, vocab_size)
labels_slice = input_ids[:, 1:].reshape(-1)
loss = F.cross_entropy(logits_slice, labels_slice, ignore_index=padding_id)
loss.backward()
```

**自检**：  
- 若 `input_ids` 没有 padding，可暂时不设 `ignore_index` 或设 `-100`。  
- 打印 `loss.item()`，应是一个正数；再打印某个参数的 `grad`，应非 None。

---

## 1.3 封装成「完整语言模型」类（DIYForCausalLM）

**目的**：把「backbone + lm_head」收进一个类，以后训练和生成都只调这个类；backbone 保持不动。

**建议顺序**：

1. **新建类**  
   类名例如 `DIYForCausalLM`，继承 `nn.Module`。

2. **`__init__`**  
   - 参数：和 `DIYModel` 一样的那一串（vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout），或先简单点，直接传这些标量。
   - 在 `__init__` 里：
     - 构造 `self.model = DIYModel(...)`（你现有的 backbone）。
     - 构造 `self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`。
   - 不要求在这一步做「权重共享」或 Config，先能跑通即可。

3. **`forward`**  
   - 参数：至少 `input_ids`，其余（如 `seq_lengths`, `past_key_values`, `use_cache`, `is_training`）可以按需传，并**原样传给** `self.model(...)`。
   - 逻辑：
     - `hidden_states = self.model(input_ids, ...)`。  
       注意：若 `self.model` 在推理且 `use_cache=True` 时返回的是 `(hidden_states, presents)`，这里要拆包，例如：
       - `if isinstance(hidden_states, tuple): hidden_states, presents = hidden_states`
       - 然后再 `logits = self.lm_head(hidden_states)`。
     - 若你希望接口统一，可以约定：训练时只返回 `logits`；推理且 `use_cache=True` 时返回 `(logits, presents)`，方便后面做生成。这一步你可以先简单点：始终只返回 `logits`，presents 先不往外返，等第三步（生成）再改也行。

4. **自检**  
   - 用 1.1 里同样的假数据，用 `DIYForCausalLM` 再跑一遍：  
     `logits = clm(input_ids, is_training=True)`，检查 `logits.shape`。  
   - 再算一次 loss 并 backward，确认和 1.2 行为一致（若 1.2 里你是用「下一个 token」的 label，这里同样用 `logits[:, :-1]` 和 `input_ids[:, 1:]`）。

**可参考的代码片段（只作提示）**：

```python
class DIYForCausalLM(nn.Module):
    def __init__(self, vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.model = DIYModel(vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, seq_lengths=None, past_key_values=None, use_cache=False, is_training=True):
        out = self.model(input_ids, seq_lengths=seq_lengths, past_key_values=past_key_values,
                         use_cache=use_cache, is_training=is_training)
        if isinstance(out, tuple):
            hidden_states, presents = out
        else:
            hidden_states, presents = out, None
        logits = self.lm_head(hidden_states)
        if presents is not None:
            return logits, presents
        return logits
```

上面只是结构示例，参数名和返回值你可以按自己的习惯改；重点是：**自己敲一遍**，遇到报错再对照 shape 和参数检查。

---

## 建议时间与自测清单

- **1.1**：约 15～20 分钟。自测：`hidden_states.shape`、`logits.shape` 符合预期。
- **1.2**：约 15～20 分钟。自测：`loss` 为正数，某参数 `.grad` 非 None。
- **1.3**：约 20～30 分钟。自测：用 `DIYForCausalLM` 得到相同 shape 的 logits，并能重复 1.2 的 loss/backward。

---

## 第一步汇总：完成 1.1 + 1.2 + 1.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 结构上**

- 有一个 **DIYForCausalLM** 类，内部包含：
  - `self.model`：现有的 DIYModel（embed → blocks → norm），输出 `hidden_states`；
  - `self.lm_head`：`nn.Linear(hidden_size, vocab_size, bias=False)`，把 hidden 映射到词表分数。
- 数据流：`input_ids` → DIYModel → `hidden_states` → lm_head → **logits**。

**2. 1.2 的体现**

- 能用 **logits** 和 **labels**（next-token：`input_ids[:, 1:]`）算 **loss**：`F.cross_entropy(logits[:, :-1].reshape(-1, V), labels.reshape(-1), ignore_index=...)`。
- 能对 loss 做 **backward**，且模型参数有 `.grad`（训练时先 `optimizer.zero_grad()` 再 `forward` 再 `optimizer.step()`）。

**3. 1.3 的体现**

- **统一入口**：训练和推理都只调 **DIYForCausalLM**，不再单独拿 DIYModel + 外面的 lm_head 拼。
- **forward** 至少支持：`input_ids`，返回 **logits**（推理且用 cache 时可按需再返回 `presents`）。
- 用同一份假数据跑「DIYForCausalLM 前向 → 取 logits → 按 1.2 的方式算 loss 并 backward」，结果和 1.2 用 DIYModel + 单独 lm_head 时一致（shape、loss 数量级、能反传）。

**4. 下一步**

- 有了上述结果，就可以进入 **第二步**：用 DIYForCausalLM 写最小训练循环、过拟合一小段数据，验证「能学」。

若某一步卡住，优先检查：  
- `DIYModel` 在 `is_training=True` 时返回的是单个 tensor 还是 tuple；  
- `logits` 与 `labels` 的 shape 是否满足 `F.cross_entropy` 的要求（N×V 与 N）。
