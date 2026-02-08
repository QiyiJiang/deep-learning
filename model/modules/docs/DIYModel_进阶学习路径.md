# DIYModel 进阶学习路径

**前提**：已完成 DIYModel 的基础学习（第一步到第五步），有可用的模型、能训练、能生成、能保存。

下面是在此基础上**深入学习 LLM 核心技术和训练流程**的学习路径，按顺序做即可由浅入深。

---

## 学习路径概览

```
阶段一：位置编码基础
  ├─ 1.1 为什么需要位置编码
  ├─ 1.2 绝对位置编码 vs 相对位置编码
  └─ 1.3 旋转位置编码（RoPE）原理与实现

阶段二：模型规模调整
  ├─ 2.1 理解模型规模的重要性
  ├─ 2.2 参数量计算
  ├─ 2.3 调整配置到 0.5B 规模
  └─ 2.4 验证模型规模

阶段三：训练流程理解
  ├─ 3.1 预训练（Pretraining）原理与实践
  ├─ 3.2 有监督微调（SFT）原理与实践
  └─ 3.3 预训练 vs SFT 的区别与联系

阶段四：模型优化与工程化
  ├─ 4.0 大规模训练前的知识补充（建议先学）
  ├─ 4.1 训练优化（Checkpoint、DDP、日志、梯度裁剪与 warmup）
  └─ 4.2 部署与推理优化（量化、KV Cache、模型保存与加载）
```

**建议顺序**：阶段一 → 阶段二 → 阶段三 → 阶段四，每个阶段内部按顺序学习。

---

## 阶段一：位置编码基础

**目标**：理解为什么需要位置编码，掌握 RoPE 的原理和实现，能在 DIYModel 中集成 RoPE。  
**前提**：已完成 DIYModel 基础学习，理解 Attention 的工作原理。  
**预计时间**：2-3 天

---

### 1.1 为什么需要位置编码

**目的**：理解 Transformer 的「置换不变性」问题，以及位置编码的作用。

**建议顺序**：

1. **理解问题**  
   - Transformer 的 self-attention 是**置换不变**的：打乱输入顺序，输出不变（除了位置信息）。
   - 但语言是有顺序的："猫吃鱼" ≠ "鱼吃猫"。
   - **问题**：没有位置信息，模型无法区分顺序。

2. **小实验验证**  
   - 用你的 DIYModel（当前没有位置编码），构造两个输入：
     - `input1 = [1, 2, 3, 4, 5]`
     - `input2 = [5, 4, 3, 2, 1]`（顺序相反）
   - 分别 forward，观察 `hidden_states` 是否相同（可能相同，因为只有 embedding，没有位置信息）。
   - **结论**：需要位置编码来区分顺序。

3. **位置编码的作用**  
   - 给每个位置一个**唯一标识**，让模型知道 token 的顺序。
   - 方式：可以是 learnable embedding、固定公式、或 RoPE（相对位置）。

**学习时间**：约 30 分钟

---

### 1.2 绝对位置编码 vs 相对位置编码

**目的**：理解两种位置编码方式的区别，以及为什么 RoPE（相对位置）更适合长序列。

**建议顺序**：

1. **绝对位置编码**  
   - 方式：每个位置有一个独立的 embedding，`pos_emb[0]`, `pos_emb[1]`, ...
   - 实现：`pos_embedding = nn.Embedding(max_seq_len, hidden_size)`
   - 应用：`hidden = token_emb + pos_embedding(position_ids)`
   - **缺点**：训练时只见过固定长度（如 512），推理时无法外推到更长序列（没见过的位置没有 embedding）。

2. **相对位置编码**  
   - 方式：编码的是「位置之间的相对关系」，而不是绝对位置。
   - 例如：位置 5 和位置 3 的相对距离是 2，位置 1005 和位置 1003 的相对距离也是 2。
   - **优点**：可以外推到更长序列（相对关系不变）。

3. **对比实验（可选）**  
   - 用绝对位置编码训练一个模型（max_seq_len=128），在 128 和 256 长度上分别测试。
   - 观察：256 长度时效果可能变差（因为位置 128-255 的 embedding 是随机初始化的，没训练过）。

**学习时间**：约 1 小时

---

### 1.3 旋转位置编码（RoPE）原理与实现

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
   - 输入：`dim`（head_dim）、`end`（最大位置）、`rope_base`（基础频率，如 1e6）。
   - 步骤：
     1. 计算基础频率：`freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2) / dim))`
     2. 计算所有位置的频率矩阵：`freqs = torch.outer(torch.arange(end), freqs)`
     3. 生成 cos/sin：`freqs_cos = torch.cos(freqs)`，`freqs_sin = torch.sin(freqs)`
     4. 重复一次以匹配完整维度：`freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)`
   - 返回：`(freqs_cos, freqs_sin)`，shape 都是 `(max_seq_len, head_dim)`

3. **实现 apply_rotary_pos_emb**  
   - 输入：`q, k`（shape `(batch, num_heads, seq_len, head_dim)`）、`cos, sin`（shape `(seq_len, head_dim)`）。
   - 步骤：
     1. 定义 `rotate_half`：把后一半维度取负并与前一半拼接：`[-x[d/2:], x[:d/2]]`
     2. 应用旋转：`q_rot = q * cos + rotate_half(q) * sin`
     3. 对 k 同样操作
   - 返回：`(q_rot, k_rot)`

4. **在 DIYModel 中集成**  
   - 在 `DIYModel.__init__` 里：
     - 预计算 `freqs_cos, freqs_sin`（用 `precompute_freqs_cis`）
     - 用 `register_buffer` 注册为 buffer（不参与训练，但会随模型保存）
   - 在 Attention 里：
     - 计算 Q/K 后，应用 RoPE：`q, k = apply_rotary_pos_emb(q, k, cos, sin)`
     - 然后再做 attention

5. **验证**  
   - 训练时用短序列（如 128），推理时用长序列（如 256），观察是否还能正常工作。
   - 对比：有/无 RoPE 在长序列上的效果差异。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6):
    """预计算 RoPE 的 cos/sin 值。"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """对 Q/K 应用旋转位置编码。"""
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    q_rot = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
    k_rot = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    return q_rot, k_rot
```

**学习时间**：约 1-2 天（包括实现和调试）

---

### 阶段一汇总：完成 1.1 + 1.2 + 1.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 理解位置编码的必要性**

- 能解释为什么 Transformer 需要位置编码（置换不变性问题）。
- 理解绝对位置编码和相对位置编码的区别。

**2. RoPE 实现**

- 有 `precompute_freqs_cis` 函数，能预计算所有位置的 cos/sin 值。
- 有 `apply_rotary_pos_emb` 函数，能对 Q/K 应用旋转。
- `DIYModel` 中集成了 RoPE，在 Attention 里正确应用。

**3. 验证外推能力**

- 训练时用短序列（如 128），推理时用长序列（如 256），模型仍能正常工作。
- 理解为什么 RoPE 能外推（相对位置关系不变）。

**4. 下一步**

- 有了 RoPE，模型就有了位置信息，可以进入 **阶段二**：调整模型规模，让它能真正学到语义信息。

---

## 阶段二：模型规模调整

**目标**：理解模型规模的重要性，掌握参数量计算，将模型从小规模（学习用）调整到 0.5B 规模（能真正学到语义信息）。  
**前提**：已完成阶段一（RoPE），理解位置编码的作用。  
**预计时间**：1-2 天

---

### 2.1 理解模型规模的重要性

**目的**：理解为什么需要调整模型规模，以及不同规模模型的能力差异。

**建议顺序**：

1. **理解当前模型的问题**  
   - **当前配置**：`hidden_size=256, num_layers=2, vocab_size=6400`
   - **参数量**：约 1-2M（非常小）
   - **问题**：
     - 模型容量太小，无法学习复杂的语义关系
     - 只能记住简单的模式，无法理解深层语义
     - 训练后效果差，生成文本质量低

2. **理解模型规模与能力的关系**  
   - **小模型（<10M）**：只能学习简单的模式，适合学习 Transformer 原理
   - **中等模型（100M-1B）**：能学习语义信息，适合实际应用
   - **大模型（>1B）**：能学习复杂知识，但训练成本高
   - **0.5B 规模**：平衡点，既能学到语义信息，又不会太大

3. **理解 Scaling Law**  
   - **参数量**：决定模型的「记忆容量」
   - **数据量**：决定模型的「知识广度」
   - **训练步数**：决定模型的「学习深度」
   - **三者平衡**：参数量、数据量、训练步数需要匹配

**学习时间**：约 30 分钟

---

### 2.2 参数量计算

**目的**：掌握如何计算模型的参数量，理解各组件对参数量的贡献。

**建议顺序**：

1. **理解参数量计算公式**  
   - **Embedding 层**：`vocab_size × hidden_size`（与 LM Head 共享权重）
   - **Attention 层**（每层）：
     - Q/K/V 投影：`3 × hidden_size × hidden_size`
     - O 投影：`hidden_size × hidden_size`
     - 总计：`4 × hidden_size²`
   - **FFN 层**（每层）：
     - Gate/Up：`2 × hidden_size × intermediate_size`
     - Down：`intermediate_size × hidden_size`
     - 总计：`3 × hidden_size × intermediate_size`
   - **Norm 层**（每层）：`2 × hidden_size`（attention_norm + feedforward_norm）
   - **LM Head**：`vocab_size × hidden_size`（与 Embedding 共享权重）

2. **实现参数量计算函数**  
   - **方法**：遍历模型的所有参数，统计总数
   - **实现**：
     ```python
     def count_parameters(model):
         """计算模型参数量（单位：M）。"""
         total = sum(p.numel() for p in model.parameters())
         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
         return total / 1e6, trainable / 1e6
     
     # 使用
     total_params, trainable_params = count_parameters(model)
     print(f"Total parameters: {total_params:.2f}M")
     print(f"Trainable parameters: {trainable_params:.2f}M")
     ```

3. **手动计算验证**  
   - **当前模型**（hidden_size=256, num_layers=2, vocab_size=6400, intermediate_size=512）：
     - Embedding：`6400 × 256 = 1.64M`
     - 每层 Attention：`4 × 256² = 0.26M`
     - 每层 FFN：`3 × 256 × 512 = 0.39M`
     - 每层 Norm：`2 × 256 = 0.0005M`
     - 2 层总计：`2 × (0.26 + 0.39 + 0.0005) = 1.3M`
     - **总参数量**：`1.64 + 1.3 = 2.94M`（约 3M）

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
def count_parameters(model):
    """计算模型参数量（单位：M）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

# 使用
model = DIYForCausalLM(config)
total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:.2f}M")
print(f"Trainable parameters: {trainable_params:.2f}M")
```

**学习时间**：约 1 小时

---

### 2.3 调整配置到 0.5B 规模

**目的**：掌握如何调整模型配置，将参数量提升到 0.5B（500M）。

**建议顺序**：

1. **理解配置参数的影响**  
   - **`hidden_size`**：影响每层的参数量（平方关系）
   - **`num_layers`**：影响层数（线性关系）
   - **`intermediate_size`**：影响 FFN 的参数量（通常设为 `4 × hidden_size`）
   - **`vocab_size`**：影响 Embedding 和 LM Head 的参数量（通常固定）

2. **设计 0.5B 配置**  
   - **目标**：参数量 ≈ 500M
   - **策略**：
     - 保持 `vocab_size=6400`（不变）
     - 增加 `hidden_size`（如 768 或 1024）
     - 增加 `num_layers`（如 12 或 16）
     - `intermediate_size = 4 × hidden_size`（标准配置）
   - **示例配置**：
     ```python
     # 配置 1：hidden_size=768, num_layers=12
     config = DIYCofig(
         vocab_size=6400,
         hidden_size=768,
         num_layers=12,
         num_heads=12,  # 通常 hidden_size % num_heads == 0
         intermediate_size=3072,  # 4 × 768
         max_seq_len=2048,
         dropout=0.1,
         eps=1e-5
     )
     # 预计参数量：约 400-500M
     
     # 配置 2：hidden_size=1024, num_layers=8
     config = DIYCofig(
         vocab_size=6400,
         hidden_size=1024,
         num_layers=8,
         num_heads=16,
         intermediate_size=4096,  # 4 × 1024
         max_seq_len=2048,
         dropout=0.1,
         eps=1e-5
     )
     # 预计参数量：约 500-600M
     ```

3. **调整模型配置**  
   - **步骤**：
     1. 修改 `DIYCofig` 的默认值，或创建新的配置实例
     2. 用新配置初始化模型：`model = DIYForCausalLM(new_config)`
     3. 验证参数量：`count_parameters(model)`
     4. 如果参数量不对，调整 `hidden_size` 或 `num_layers`

4. **注意显存限制**  
   - **问题**：模型变大后，显存占用增加
   - **计算**：显存 ≈ 参数量 × 4 bytes（fp32）或 2 bytes（fp16）
   - **0.5B 模型**：约 2GB（fp32）或 1GB（fp16）
   - **训练时**：还需要额外的显存（梯度、优化器状态、激活值）
   - **建议**：如果显存不足，可以：
     - 使用混合精度训练（fp16）
     - 减小 batch_size
     - 使用 gradient checkpointing

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 创建 0.5B 配置
config_05b = DIYCofig(
    vocab_size=6400,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    intermediate_size=3072,  # 4 × hidden_size
    max_seq_len=2048,
    dropout=0.1,
    eps=1e-5
)

# 初始化模型
model = DIYForCausalLM(config_05b)

# 验证参数量
total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:.2f}M")
assert 400 <= total_params <= 600, f"参数量 {total_params:.2f}M 不在 400-600M 范围内"
```

**学习时间**：约 2-3 小时（包括调试）

---

### 2.4 验证模型规模

**目的**：验证模型规模调整是否成功，确保模型能正常工作。

**建议顺序**：

1. **验证参数量**  
   - **方法**：用 `count_parameters` 函数计算
   - **目标**：参数量在 400-600M 范围内（接近 0.5B）
   - **如果不对**：调整 `hidden_size` 或 `num_layers`

2. **验证模型能正常前向传播**  
   - **方法**：构造一个小的输入，测试 forward
   - **实现**：
     ```python
     # 测试前向传播
     batch_size, seq_len = 2, 128
     input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
     
     model.eval()
     with torch.no_grad():
         logits = model(input_ids)
         print(f"Logits shape: {logits.shape}")  # 应该是 (batch_size, seq_len, vocab_size)
     ```

3. **验证模型能正常训练**  
   - **方法**：运行一个训练步骤，检查 loss
   - **实现**：
     ```python
     # 测试训练步骤
     model.train()
     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
     
     input_ids = torch.randint(0, config.vocab_size, (2, 128))
     labels = input_ids[:, 1:]
     
     logits = model(input_ids)
     loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), labels.reshape(-1))
     loss.backward()
     optimizer.step()
     
     print(f"Loss: {loss.item():.4f}")
     ```

4. **对比小模型和大模型**  
   - **小模型**（当前）：参数量小，训练快，但效果差
   - **大模型**（0.5B）：参数量大，训练慢，但效果好
   - **理解**：模型规模是性能的基础，没有足够的参数量，再好的训练方法也无法提升效果

**学习时间**：约 1 小时

---

### 阶段二汇总：完成 2.1 + 2.2 + 2.3 + 2.4 后的结果

做完上面四小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 理解模型规模的重要性**

- 能解释为什么需要调整模型规模（小模型无法学到语义信息）。
- 理解模型规模与能力的关系（参数量决定记忆容量）。

**2. 参数量计算**

- 能计算模型的参数量（手动计算和代码计算）。
- 理解各组件对参数量的贡献（Embedding、Attention、FFN、Norm）。

**3. 调整配置到 0.5B**

- 能设计 0.5B 配置（hidden_size、num_layers、intermediate_size）。
- 能调整模型配置，参数量在 400-600M 范围内。

**4. 验证模型规模**

- 能验证参数量是否正确。
- 能验证模型能正常前向传播和训练。

**5. 下一步**

- 有了 0.5B 规模的模型，可以进入 **阶段三**：理解预训练和 SFT 的训练流程。

---

## 阶段三：训练流程理解

**目标**：理解预训练和 SFT 的区别、各自的作用、以及如何实现。  
**前提**：已完成阶段二（模型规模调整），有 0.5B 规模的模型。  
**预计时间**：3-4 天

---

### 3.1 预训练（Pretraining）原理与实践

**目的**：理解预训练的目标、数据格式、以及如何实现一个简单的预训练脚本。

**建议顺序**：

1. **理解预训练的目标**  
   - **目标**：让模型学习「语言的基本规律」和「世界知识」
   - **任务**：Next-token prediction（给定前文，预测下一个 token）
   - **特点**：
     - 无监督学习（不需要人工标注）
     - 数据量大（TB 级别）
     - 训练时间长（可能需要数周）
     - 学习的是「通用知识」

2. **理解预训练的数据格式**  
   - 格式：`{"text": "连续的一段文本..."}`
   - 特点：文本是**连续的**，没有结构化的问答对
   - 处理：按固定长度（如 512）切分，构造 `input_ids` 和 `labels`（labels = input_ids 右移一位）

3. **实现简单的预训练脚本**  
   - **数据加载**：
     - 读取 jsonl 文件，每行是一个 `{"text": "..."}`
     - 用 Tokenizer 编码成 `input_ids`
     - 把所有 `input_ids` 拼接，按固定长度切分
     - 构造 `labels = input_ids[:, 1:]`（next-token）
   - **训练循环**：
     - `logits = model(input_ids)`
     - `loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels.reshape(-1))`
     - `loss.backward()` → `optimizer.step()`
   - **观察**：loss 下降，模型能「续写」文本（虽然质量可能不高）

4. **理解预训练的作用**  
   - 预训练让模型「记住」了大量文本中的模式：
     - 语法规则（主谓宾、时态等）
     - 常识知识（"北京是中国的首都"）
     - 语言风格（正式/非正式、不同领域）
   - 但预训练后的模型**不会对话**：它只会「续写」，不知道什么时候该「回答」

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 数据加载
def load_pretrain_data(file_path, tokenizer, max_seq_len=512):
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    
    # 编码所有文本
    all_input_ids = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt")
        all_input_ids.extend(encoded['input_ids'][0].tolist())
    
    # 按 max_seq_len 切分
    input_ids_list = []
    for i in range(0, len(all_input_ids), max_seq_len):
        chunk = all_input_ids[i:i+max_seq_len]
        if len(chunk) == max_seq_len:
            input_ids_list.append(chunk)
    return input_ids_list

# 训练循环
for step, input_ids in enumerate(loader):
    input_ids = torch.tensor(input_ids).to(device)
    labels = input_ids[:, 1:]  # next-token
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels.reshape(-1))
    loss.backward()
    optimizer.step()
```

**学习时间**：约 1-1.5 天

---

### 3.2 有监督微调（SFT）原理与实践

**目的**：理解 SFT 的目标、数据格式、以及如何实现一个简单的 SFT 脚本。

**建议顺序**：

1. **理解 SFT 的目标**  
   - **目标**：让模型学习「如何与人对话」或「如何遵循指令」
   - **任务**：Instruction following（给定指令，生成回答）
   - **特点**：
     - 有监督学习（需要人工标注的问答对）
     - 数据量相对小（GB 级别）
     - 训练时间短（几小时到几天）
     - 学习的是「对话格式」和「指令理解」

2. **理解 SFT 的数据格式**  
   - 格式：`{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
   - 特点：结构化的对话，有明确的「问题」和「回答」
   - 处理：
     - 用 Tokenizer 的 `apply_chat_template` 转换成文本格式
     - 编码成 `input_ids`
     - 构造 `loss_mask`（只对 assistant 部分算 loss，user 部分用 `ignore_index`）

3. **实现简单的 SFT 脚本**  
   - **数据加载**：
     - 读取 jsonl 文件，每行是一个对话
     - 用 `tokenizer.apply_chat_template` 转换成文本格式
     - 编码成 `input_ids`
     - 构造 `loss_mask`（1 表示算 loss，0 表示忽略）
   - **训练循环**：
     - `logits = model(input_ids)`
     - `loss = F.cross_entropy(..., ignore_index=padding_id)`，只对 assistant 部分算 loss
     - `loss.backward()` → `optimizer.step()`
   - **观察**：loss 下降，模型能「回答问题」或「遵循指令」

4. **理解 SFT 的作用**  
   - SFT 让模型「学会」对话格式：
     - 识别「问题」和「回答」的结构
     - 知道什么时候该「回答」，什么时候该「续写」
     - 学习「指令理解」（如 "翻译"、"总结" 等）
   - 但 SFT **不能补充知识**：如果预训练时没学过的知识，SFT 也学不会

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 数据加载
def load_sft_data(file_path, tokenizer):
    conversations = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            conversations.append(data['conversations'])
    
    input_ids_list = []
    loss_mask_list = []
    for conv in conversations:
        # 转换成文本格式
        text = tokenizer.apply_chat_template(conv, tokenize=False)
        # 编码
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded['input_ids'][0]
        
        # 构造 loss_mask（只对 assistant 部分算 loss）
        loss_mask = construct_loss_mask(input_ids, tokenizer)
        
        input_ids_list.append(input_ids.tolist())
        loss_mask_list.append(loss_mask.tolist())
    
    return input_ids_list, loss_mask_list

# 训练循环
for step, (input_ids, loss_mask) in enumerate(loader):
    input_ids = torch.tensor(input_ids).to(device)
    loss_mask = torch.tensor(loss_mask).to(device)
    
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, V), input_ids[:, 1:].reshape(-1), 
                           ignore_index=tokenizer.pad_token_id)
    loss = (loss.view(loss_mask.shape) * loss_mask).sum() / loss_mask.sum()
    loss.backward()
    optimizer.step()
```

**学习时间**：约 1-1.5 天

---

### 3.3 预训练 vs SFT 的区别与联系

**目的**：深入理解两者的区别、联系、以及为什么需要两步训练。

**建议顺序**：

1. **对比两者的区别**  
   - **数据**：
     - 预训练：大规模无标注文本（Wikipedia、新闻、书籍）
     - SFT：高质量的对话数据（需要人工构造或筛选）
   - **任务**：
     - 预训练：Next-token prediction（续写）
     - SFT：Instruction following（回答问题）
   - **训练方式**：
     - 预训练：无监督，所有位置都算 loss
     - SFT：有监督，只对回答部分算 loss
   - **作用**：
     - 预训练：学「知识」
     - SFT：学「格式」

2. **理解两者的联系**  
   - SFT 通常**基于预训练模型**：先用预训练模型初始化，再在对话数据上微调
   - 预训练是「打基础」，SFT 是「调格式」
   - 两者缺一不可：
     - 只有预训练 → 只会续写，不会对话
     - 只有 SFT → 没有知识基础，回答质量差

3. **对比实验**  
   - **实验 1**：只用预训练
     - 训练一个模型（只用预训练数据）
     - 测试：输入一个问题，观察输出（通常只会续写，不会回答）
   - **实验 2**：预训练 + SFT
     - 先用预训练数据训练，再用 SFT 数据微调
     - 测试：输入一个问题，观察输出（应该能回答问题）
   - **结论**：需要两步训练，不能一步到位

4. **理解为什么不能一步到位**  
   - 如果只用 SFT 数据训练（没有预训练）：
     - 数据量太小，模型学不到足够的「知识」
     - 模型只会「格式」，但回答质量差（没有知识基础）
   - 如果只用预训练数据训练（没有 SFT）：
     - 模型有「知识」，但不会「对话」
     - 输入问题后，模型只会续写，不会识别「这是问题，需要回答」

**学习时间**：约 1 天

---

### 阶段三汇总：完成 3.1 + 3.2 + 3.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 理解预训练**

- 能解释预训练的目标（学习语言规律和世界知识）。
- 理解预训练的数据格式（连续文本）和任务（next-token prediction）。
- 能实现简单的预训练脚本，模型能「续写」文本。

**2. 理解 SFT**

- 能解释 SFT 的目标（学习对话格式和指令理解）。
- 理解 SFT 的数据格式（结构化对话）和任务（instruction following）。
- 能实现简单的 SFT 脚本，模型能「回答问题」。

**3. 理解两者的区别与联系**

- 能清晰对比预训练和 SFT 的区别（数据、任务、作用）。
- 理解为什么需要两步训练（预训练打基础，SFT 调格式）。
- 能解释为什么不能一步到位（只有预训练不会对话，只有 SFT 没有知识基础）。

**4. 下一步**

- 理解了训练流程，可以进入 **阶段四**：模型优化与工程化。建议顺序：**先 4.0 大规模训练前的知识补充 → 4.1 训练优化 → 4.2 部署与推理优化**。详细内容见《DIYModel_进阶学习路径_阶段4.md》。

---

## 阶段四：模型优化与工程化

**目标**：先做好**训练优化**（可恢复、稳定、可观测），再做**部署与推理优化**，使模型真正可工程化使用。  
**前提**：已完成阶段三（预训练和 SFT），有小规模数据和训练脚本可跑。  
**预计时间**：4–6 天

**学习顺序**：  
- **4.0 大规模训练前的知识补充**（建议先学）：checkpoint 与恢复、warmup、梯度裁剪、日志等，避免长训翻车。  
- **4.1 训练优化**：Checkpoint 管理、DDP、日志监控、数据加载衔接、梯度累积与混合精度（可选）。  
- **4.2 部署与推理优化**：模型保存与加载、量化、KV Cache、Batch 推理。

详细小节说明、代码提示与自检见 **《DIYModel_进阶学习路径_阶段4.md》**。

---

### 阶段四汇总：完成所有优化后的结果

做完上面所有小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 大规模训练前（4.0）**

- 理解并会使用 checkpoint 恢复、warmup、梯度裁剪、基础日志，为长训打基础。

**2. 训练优化（4.1）**

- 能实现完整 checkpoint 管理（保存与恢复），能实现 DDP 多卡训练。
- 能实现日志与监控；数据加载与阶段三流式/分片方案衔接良好。

**3. 部署与推理优化（4.2）**

- 能区分训练用 checkpoint 与推理用权重，并正确保存与加载。
- 理解量化、KV Cache、batch 推理的作用，并能做简单推理优化。

**4. 完整工程化**

- 训练脚本具备可恢复、可观测、可扩展；推理侧具备基本优化与部署能力。

**4. 总结**

完成所有四个阶段的学习后，你应该：

1. **理解位置编码**：知道为什么需要位置编码，掌握 RoPE 的原理和实现。
2. **理解模型规模**：知道为什么需要调整模型规模，掌握参数量计算和配置调整。
3. **理解训练流程**：知道预训练和 SFT 的区别与联系，能实现两种训练。
4. **掌握推理优化**：知道如何加速推理，减少显存占用。
5. **掌握工程化实践**：知道如何管理 checkpoint、分布式训练、日志监控等。

**恭喜！你已经掌握了 LLM 的核心技术和工程化实践！**

---

## 学习顺序建议

| 阶段 | 核心内容 | 预计时间 | 优先级 |
|------|----------|----------|--------|
| 一 | RoPE 原理与实现 | 2-3 天 | P0（基础） |
| 二 | 模型规模调整到 0.5B | 1-2 天 | P0（核心） |
| 三 | 预训练 vs SFT | 3-4 天 | P0（核心） |
| 四 | 大规模训练前补充（4.0） | 约 1 天 | P1（建议先学） |
| 四 | 训练优化（4.1） | 约 2 天 | P1（重要） |
| 四 | 部署与推理优化（4.2） | 1-2 天 | P1（重要） |

**总预计时间**：约 2-3 周（按每天 2-3 小时计算）

---

## 学习建议

1. **先理解原理，再动手实现**：每个阶段先看原理，理解「为什么」，再写代码。
2. **小步快跑**：每个阶段都先做最小实现，验证能跑通，再完善。
3. **对比实验**：多做对比（有/无 RoPE、小模型 vs 大模型、预训练 vs SFT、有/无优化），理解差异。
4. **记录笔记**：记录每个阶段的收获、遇到的问题、解决方案。

完成这四个阶段后，你就有了一个**完整的、可用的、工程化的语言模型**，可以在此基础上继续扩展（更大的模型、更多数据、更多训练技巧等）。
