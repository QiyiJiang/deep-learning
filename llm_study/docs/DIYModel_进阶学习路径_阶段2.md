# 阶段二详细学习计划：模型规模调整

**目标**：理解模型规模的重要性，掌握参数量计算，将模型从小规模（学习用）调整到 0.5B 规模（能真正学到语义信息）。  
**前提**：已完成阶段一（RoPE），理解位置编码的作用。  
**预计时间**：1-2 天

**建议**：新建一个单独的练习脚本（例如 `step_model_scaling.py`），不要直接改 `DIYModel.py`，方便对比和回滚。

---

## 2.1 理解模型规模的重要性

**目的**：理解为什么需要调整模型规模，以及不同规模模型的能力差异。

**建议顺序**：

1. **理解当前模型的问题**  
   - **当前配置**：`hidden_size=256, num_layers=2, vocab_size=6400, intermediate_size=512`
   - **参数量**：约 1-2M（非常小）
   - **问题**：
     - 模型容量太小，无法学习复杂的语义关系
     - 只能记住简单的模式，无法理解深层语义
     - 训练后效果差，生成文本质量低
     - 无法真正"理解"语言，只能做简单的模式匹配

2. **理解模型规模与能力的关系**  
   - **小模型（<10M）**：
     - 只能学习简单的模式（如字符级别的统计规律）
     - 适合学习 Transformer 原理和调试
     - 无法学习语义信息
   - **中等模型（100M-1B）**：
     - 能学习语义信息（词义、语法、常识）
     - 适合实际应用（对话、文本生成）
     - 0.5B 是一个很好的平衡点
   - **大模型（>1B）**：
     - 能学习复杂知识（推理、多步思考）
     - 训练成本高，需要大量资源

3. **理解 Scaling Law**  
   - **参数量**：决定模型的「记忆容量」
     - 更多参数 = 能记住更多模式
     - 但参数太多 = 训练困难、容易过拟合
   - **数据量**：决定模型的「知识广度」
     - 更多数据 = 覆盖更多领域
     - 但数据太多 = 训练时间长
   - **训练步数**：决定模型的「学习深度」
     - 更多步数 = 学得更深入
     - 但步数太多 = 可能过拟合
   - **三者平衡**：参数量、数据量、训练步数需要匹配
     - 小模型 + 大数据 = 浪费数据
     - 大模型 + 小数据 = 容易过拟合

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 查看当前模型的配置
from llm_study import DIYConfig, DIYForCausalLM

config = DIYConfig()  # 使用默认配置
print("当前配置：")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_layers: {config.num_layers}")
print(f"  vocab_size: {config.vocab_size}")
print(f"  intermediate_size: {config.intermediate_size}")

# 创建模型并观察效果（可选：简单测试生成质量）
model = DIYForCausalLM(config)
# 可以简单测试一下生成效果，观察是否"胡言乱语"
```

**自检**：
- 能解释为什么需要调整模型规模（小模型无法学到语义信息）。
- 理解模型规模与能力的关系（参数量决定记忆容量）。
- 理解 Scaling Law 的基本概念（参数量、数据量、训练步数的平衡）。

**学习时间**：约 30 分钟

---

## 2.2 参数量计算

**目的**：掌握如何计算模型的参数量，理解各组件对参数量的贡献。

**建议顺序**：

1. **理解参数量计算公式**  
   - **Embedding 层**：`vocab_size × hidden_size`
     - 注意：与 LM Head 共享权重（weight tying），所以只算一次
   - **Attention 层**（每层）：
     - Q/K/V 投影：`3 × hidden_size × hidden_size`（fused QKV）
     - O 投影：`hidden_size × hidden_size`
     - 总计：`4 × hidden_size²`
   - **FFN 层**（每层）：
     - Gate：`hidden_size × intermediate_size`
     - Up：`hidden_size × intermediate_size`
     - Down：`intermediate_size × hidden_size`
     - 总计：`3 × hidden_size × intermediate_size`
   - **Norm 层**（每层）：`2 × hidden_size`
     - attention_norm：`hidden_size`
     - feedforward_norm：`hidden_size`
   - **LM Head**：`vocab_size × hidden_size`（与 Embedding 共享权重，不算）

2. **实现参数量计算函数**  
   - **方法**：遍历模型的所有参数，统计总数
   - **实现**：
     ```python
     def count_parameters(model):
         """计算模型参数量（单位：M）。"""
         total = sum(p.numel() for p in model.parameters())
         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
         return total / 1e6, trainable / 1e6
     ```
   - **使用**：创建模型后调用，验证参数量

3. **手动计算验证**  
   - **当前模型**（hidden_size=256, num_layers=2, vocab_size=6400, intermediate_size=512）：
     - Embedding：`6400 × 256 = 1,638,400` ≈ 1.64M
     - 每层 Attention：`4 × 256² = 262,144` ≈ 0.26M
     - 每层 FFN：`3 × 256 × 512 = 393,216` ≈ 0.39M
     - 每层 Norm：`2 × 256 = 512` ≈ 0.0005M
     - 2 层总计：`2 × (0.26 + 0.39 + 0.0005) = 1.301M`
     - **总参数量**：`1.64 + 1.301 = 2.941M` ≈ 3M
   - **验证**：用代码计算，应该接近 3M

4. **理解各组件对参数量的贡献**  
   - **Embedding 占比**：在小模型中占比高（如 3M 模型中占 1.64M，约 55%）
   - **Attention 占比**：随 hidden_size 平方增长
   - **FFN 占比**：通常最大（因为 intermediate_size = 4 × hidden_size）
   - **Norm 占比**：很小，可以忽略

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
def count_parameters(model):
    """计算模型参数量（单位：M）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

# 使用
from llm_study import DIYForCausalLM, DIYConfig

config = DIYConfig()
model = DIYForCausalLM(config)

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:.2f}M")
print(f"Trainable parameters: {trainable_params:.2f}M")

# 手动计算验证
vocab_size = config.vocab_size
hidden_size = config.hidden_size
num_layers = config.num_layers
intermediate_size = config.intermediate_size

embedding_params = vocab_size * hidden_size / 1e6
attention_params_per_layer = 4 * hidden_size * hidden_size / 1e6
ffn_params_per_layer = 3 * hidden_size * intermediate_size / 1e6
norm_params_per_layer = 2 * hidden_size / 1e6

total_manual = embedding_params + num_layers * (attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer)
print(f"\n手动计算参数量: {total_manual:.2f}M")
print(f"代码计算参数量: {total_params:.2f}M")
print(f"差异: {abs(total_manual - total_params):.4f}M")
```

**自检**：
- 能手动计算模型的参数量（至少能估算）。
- 能实现参数量计算函数，代码计算和手动计算结果接近。
- 理解各组件对参数量的贡献（Embedding、Attention、FFN、Norm）。

**常见问题**：
- **Q: 为什么代码计算和手动计算有差异？**  
  A: 可能是因为 weight tying（Embedding 和 LM Head 共享权重），代码只算一次，手动计算可能算了两次。检查一下 `lm_head.weight` 和 `embed_tokens.weight` 是否是同一个对象。
- **Q: 如何查看每个组件的参数量？**  
  A: 可以用 `model.named_parameters()` 遍历，按模块名称分组统计。

**学习时间**：约 1 小时

---

## 2.3 调整配置到 0.5B 规模

**目的**：掌握如何调整模型配置，将参数量提升到 0.5B（500M）。

**建议顺序**：

1. **理解配置参数的影响**  
   - **`hidden_size`**：影响每层的参数量（平方关系）
     - Attention：`4 × hidden_size²`（平方）
     - FFN：`3 × hidden_size × intermediate_size`（线性，但 intermediate_size 通常 = 4 × hidden_size）
     - 影响最大
   - **`num_layers`**：影响层数（线性关系）
     - 每层参数 = Attention + FFN + Norm
     - 影响中等
   - **`intermediate_size`**：影响 FFN 的参数量（通常设为 `4 × hidden_size`）
     - FFN：`3 × hidden_size × intermediate_size`
     - 如果设为 `4 × hidden_size`，则 FFN = `12 × hidden_size²`
   - **`vocab_size`**：影响 Embedding 和 LM Head 的参数量（通常固定）
     - Embedding：`vocab_size × hidden_size`
     - 如果 vocab_size 很大，占比会很高

2. **设计 0.5B 配置**  
   - **目标**：参数量 ≈ 500M
   - **策略**：
     - 保持 `vocab_size=6400`（不变）
     - 增加 `hidden_size`（如 768 或 1024）
     - 增加 `num_layers`（需要更多层才能达到 500M）
     - `intermediate_size = 4 × hidden_size`（标准配置）
     - `num_heads` 需要能被 `hidden_size` 整除（如 hidden_size=768，num_heads=12）
   - **参数量估算**：
     - 每层参数量 ≈ `4 × hidden_size² + 12 × hidden_size² + 2 × hidden_size` ≈ `16 × hidden_size²`
     - hidden_size=768：每层约 9.4M，需要约 52 层才能达到 500M
     - hidden_size=1024：每层约 16.8M，需要约 29 层才能达到 500M
   - **示例配置 1**（推荐，hidden_size=1024）：
     ```python
     config_05b_v1 = DIYConfig(
         vocab_size=6400,
         hidden_size=1024,
         num_layers=29,  # 约 493M
         num_heads=16,  # 1024 % 16 == 0
         intermediate_size=4096,  # 4 × 1024
         max_seq_len=2048,
         dropout=0.1,
         eps=1e-5
     )
     # 预计参数量：约 493M
     ```
   - **示例配置 2**（hidden_size=768，层数较多）：
     ```python
     config_05b_v2 = DIYConfig(
         vocab_size=6400,
         hidden_size=768,
         num_layers=24,  # 约 231M（如果显存有限可以用这个）
         num_heads=12,  # 768 % 12 == 0
         intermediate_size=3072,  # 4 × 768
         max_seq_len=2048,
         dropout=0.1,
         eps=1e-5
     )
     # 预计参数量：约 231M（如果显存有限，可以用这个配置）
     ```
   - **注意**：要达到 500M，需要较多的层数。如果显存有限，可以先从较小的配置开始（如 200-300M），逐步增加。

3. **调整模型配置**  
   - **步骤**：
     1. 创建新的配置实例（用上面的示例配置）
     2. 用新配置初始化模型：`model = DIYForCausalLM(new_config)`
     3. 验证参数量：`count_parameters(model)`
     4. 如果参数量不对，调整 `hidden_size` 或 `num_layers`
   - **注意**：
     - `num_heads` 必须能被 `hidden_size` 整除
     - `intermediate_size` 通常设为 `4 × hidden_size`（这是标准配置）
     - `max_seq_len` 可以增大（如 2048），但要确保 `freqs_cos` 和 `freqs_sin` 也足够大

4. **注意显存限制**  
   - **问题**：模型变大后，显存占用增加
   - **计算**：
     - 参数量（fp32）：`参数量 × 4 bytes`
     - 参数量（fp16）：`参数量 × 2 bytes`
     - 0.5B 模型（fp32）：约 2GB
     - 0.5B 模型（fp16）：约 1GB
   - **训练时**：还需要额外的显存
     - 梯度：`参数量 × 4 bytes`（fp32）
     - 优化器状态：`参数量 × 8 bytes`（AdamW）
     - 激活值：取决于 batch_size 和 seq_len
   - **建议**：
     - 如果显存不足，可以：
       - 使用混合精度训练（fp16/bfloat16）
       - 减小 batch_size
       - 减小 max_seq_len
       - 使用 gradient checkpointing

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
from llm_study import DIYForCausalLM, DIYConfig

# 创建 0.5B 配置
config_05b = DIYConfig(
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
# 注意：实际参数量可能比预期小，需要根据实际情况调整层数
# 要达到 500M，hidden_size=1024 需要约 29 层，hidden_size=768 需要约 52 层
# 如果显存有限，可以从较小的配置开始（如 200-300M）

# 估算显存占用（fp32）
memory_fp32 = total_params * 4 / 1024  # GB
memory_fp16 = total_params * 2 / 1024  # GB
print(f"\n显存占用估算（仅参数）：")
print(f"  FP32: {memory_fp32:.2f} GB")
print(f"  FP16: {memory_fp16:.2f} GB")
print(f"  训练时（FP32 + 梯度 + 优化器）：约 {memory_fp32 * 3:.2f} GB")
```

**自检**：
- 能设计 0.5B 配置（hidden_size、num_layers、intermediate_size）。
- 能调整模型配置，参数量在 400-600M 范围内。
- 理解显存限制，知道如何估算显存占用。

**常见问题**：
- **Q: 如何快速估算参数量？**  
  A: 主要看 `hidden_size` 和 `num_layers`。粗略估算：
     - 每层参数量 ≈ `16 × hidden_size²`（Attention: 4×hidden_size² + FFN: 12×hidden_size²）
     - 总参数量 ≈ `vocab_size × hidden_size + num_layers × 16 × hidden_size²`
     - 例如：hidden_size=1024，每层约 16.8M；hidden_size=768，每层约 9.4M
- **Q: 为什么实际参数量比预期小？**  
  A: 学习计划中的估算有误。实际计算：
     - hidden_size=1024, num_layers=12 → 约 208M（不是 500M）
     - 要达到 500M，hidden_size=1024 需要约 29 层
     - 要达到 500M，hidden_size=768 需要约 52 层
- **Q: hidden_size 和 num_layers 哪个更重要？**  
  A: 根据 MobileLLM 论文，深度（num_layers）比宽度（hidden_size）更重要。但要注意，hidden_size 对参数量的影响是平方关系，所以增加 hidden_size 会更快增加参数量。
- **Q: 显存不足怎么办？**  
  A: 使用混合精度训练（fp16/bfloat16），减小 batch_size，使用 gradient checkpointing。如果还是不够，可以先从较小的配置开始（如 200-300M）。

**学习时间**：约 2-3 小时（包括调试）

---

## 2.4 验证模型规模

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
   - **检查**：
     - 没有报错
     - logits 的 shape 正确
     - logits 的值合理（不是 NaN 或 Inf）

3. **验证模型能正常训练**  
   - **方法**：运行一个训练步骤，检查 loss
   - **实现**：
     ```python
     # 测试训练步骤
     model.train()
     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
     
     batch_size, seq_len = 2, 128
     input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
     labels = input_ids[:, 1:]
     
     logits = model(input_ids)
     loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), labels.reshape(-1))
     loss.backward()
     optimizer.step()
     
     print(f"Loss: {loss.item():.4f}")
     print(f"Loss is finite: {torch.isfinite(loss)}")
     ```
   - **检查**：
     - 没有报错
     - loss 是有限值（不是 NaN 或 Inf）
     - loss 的值合理（通常在 6-10 之间，对于随机初始化的模型）

4. **对比小模型和大模型**  
   - **小模型**（当前，3M）：
     - 参数量小，训练快
     - 但效果差，生成文本质量低
   - **大模型**（0.5B）：
     - 参数量大，训练慢
     - 但效果好，能学到语义信息
   - **理解**：模型规模是性能的基础，没有足够的参数量，再好的训练方法也无法提升效果

5. **检查显存占用（可选）**  
   - **方法**：用 `torch.cuda.memory_allocated()` 查看显存占用
   - **实现**：
     ```python
     if torch.cuda.is_available():
         model = model.cuda()
         input_ids = input_ids.cuda()
         
         torch.cuda.reset_peak_memory_stats()
         logits = model(input_ids)
         memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
         print(f"显存占用: {memory_used:.2f} GB")
     ```

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
def verify_model(model, config):
    """验证模型是否能正常工作。"""
    model.eval()
    
    # 1. 测试前向传播
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size), \
            f"Logits shape 错误: {logits.shape}"
        assert torch.isfinite(logits).all(), "Logits 包含 NaN 或 Inf"
        print("✓ 前向传播正常")
    
    # 2. 测试训练步骤
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    labels = input_ids[:, 1:]
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), labels.reshape(-1))
    
    assert torch.isfinite(loss), "Loss 是 NaN 或 Inf"
    print(f"✓ 训练步骤正常，loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    print("✓ 反向传播正常")

# 使用
config_05b = DIYConfig(...)  # 你的 0.5B 配置
model = DIYForCausalLM(config_05b)
verify_model(model, config_05b)
```

**自检**：
- 能验证参数量是否正确（400-600M）。
- 能验证模型能正常前向传播和训练。
- 理解小模型和大模型的区别（参数量 vs 效果）。

**常见问题**：
- **Q: Loss 是 NaN？**  
  A: 可能是学习率太大，或者模型初始化有问题。尝试减小学习率（如 1e-5）。
- **Q: 显存不足？**  
  A: 减小 batch_size 或 seq_len，或者使用混合精度训练。
- **Q: 训练很慢？**  
  A: 这是正常的，0.5B 模型比 3M 模型慢很多。可以考虑使用更小的 batch_size 或更短的序列。

**学习时间**：约 1 小时

---

## 阶段二汇总：完成 2.1 + 2.2 + 2.3 + 2.4 后的结果

做完上面四小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 理解模型规模的重要性**

- 能解释为什么需要调整模型规模（小模型无法学到语义信息）。
- 理解模型规模与能力的关系（参数量决定记忆容量）。
- 理解 Scaling Law 的基本概念（参数量、数据量、训练步数的平衡）。

**2. 参数量计算**

- 能计算模型的参数量（手动计算和代码计算）。
- 理解各组件对参数量的贡献（Embedding、Attention、FFN、Norm）。
- 能实现参数量计算函数，代码计算和手动计算结果接近。

**3. 调整配置到 0.5B**

- 能设计 0.5B 配置（hidden_size、num_layers、intermediate_size）。
- 能调整模型配置，参数量接近目标（如 200-500M，根据显存情况）。
- 理解显存限制，知道如何估算显存占用。
- **注意**：要达到 500M 需要较多层数（hidden_size=1024 需要约 29 层），如果显存有限，可以从较小的配置开始。

**4. 验证模型规模**

- 能验证参数量是否正确。
- 能验证模型能正常前向传播和训练。
- 理解小模型和大模型的区别（参数量 vs 效果）。

**5. 代码质量**

- 有参数量计算函数。
- 有模型验证函数。
- 代码结构清晰，有适当的注释。

**6. 下一步**

- 有了 0.5B 规模的模型，可以进入 **阶段三**：理解预训练和 SFT 的训练流程。

---

## 学习建议

1. **先理解原理，再动手实现**：先理解为什么需要调整模型规模，再设计配置。
2. **小步快跑**：先实现参数量计算函数，验证当前模型的参数量；再设计 0.5B 配置，验证参数量；最后验证模型能正常工作。
3. **多做实验**：对比不同配置的参数量，理解各参数的影响。
4. **记录笔记**：记录参数量计算公式、不同配置的参数量、以及遇到的问题。

若某一步卡住，优先检查：
- 参数量计算公式是否正确（注意 weight tying）；
- `num_heads` 是否能被 `hidden_size` 整除；
- `intermediate_size` 是否设为 `4 × hidden_size`（标准配置）；
- 显存是否足够（0.5B 模型训练时可能需要 4-8GB 显存）。
