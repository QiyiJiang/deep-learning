# 第四步详细学习计划：配置与保存——方便复现

**目标**：把超参集中到一个 Config 类，能保存/加载模型权重（和配置），便于复现实验和继续训练。  
**前提**：已完成第三步，有可用的 `DIYForCausalLM`，且能正常训练和生成。  
**建议**：新建一个单独脚本（例如 `step4_config_save.py`），先写 Config，再试保存/加载，最后把 config 和权重一起存。

---

## 4.1 引入简单 Config

**目的**：把散落在各处的超参（vocab_size, num_layers, hidden_size 等）集中到一个类，改一处即可复现同一模型结构。

**建议顺序**：

1. **选择 Config 的实现方式**  
   - **方式 A**：用 `dataclass`（推荐，简洁）：
     ```python
     from dataclasses import dataclass
     
     @dataclass
     class DIYConfig:
         vocab_size: int = 6400
         num_layers: int = 2
         hidden_size: int = 256
         num_heads: int = 4
         max_seq_len: int = 128
         dropout: float = 0.1
     ```
   - **方式 B**：用普通类：
     ```python
     class DIYConfig:
         def __init__(self, vocab_size=6400, num_layers=2, hidden_size=256, 
                      num_heads=4, max_seq_len=128, dropout=0.1):
             self.vocab_size = vocab_size
             self.num_layers = num_layers
             self.hidden_size = hidden_size
             self.num_heads = num_heads
             self.max_seq_len = max_seq_len
             self.dropout = dropout
     ```
   - 建议用 **方式 A**（dataclass），代码更简洁，且支持类型提示。

2. **修改 DIYModel 的 __init__**  
   - 参数从 `(vocab_size, num_layers, ...)` 改为接收一个 `config: DIYConfig`。
   - 在 `__init__` 里用 `config.vocab_size`、`config.num_layers` 等来构造各层。
   - 例如：`self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)`。

3. **修改 DIYForCausalLM 的 __init__**  
   - 同样改为接收 `config: DIYConfig`，传给 `DIYModel(config)`。
   - 例如：`self.model = DIYModel(config)`。

4. **自检**  
   - 用 `config = DIYConfig()` 构造模型：`model = DIYForCausalLM(config)`。
   - 确认模型能正常 forward 和 backward（用第二步的训练循环测试）。
   - 改 `config` 的参数（如 `num_layers=4`），重新构造模型，确认结构变化。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
from dataclasses import dataclass

@dataclass
class DIYConfig:
    vocab_size: int = 6400
    num_layers: int = 2
    hidden_size: int = 256
    num_heads: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

# 在 DIYModel.__init__ 里
def __init__(self, config: DIYConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.num_layers = config.num_layers
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    # ...
```

---

## 4.2 保存与加载 state_dict

**目的**：理解 `state_dict` 是什么，如何保存/加载模型权重，以及 `strict` 参数的作用。

**建议顺序**：

1. **理解 state_dict**  
   - `model.state_dict()` 返回一个字典，key 是参数名（如 `"model.embed_tokens.weight"`），value 是对应的 tensor。
   - 只保存**可训练参数**（`nn.Parameter`），不包括 buffer（如 batch norm 的 running_mean）或非参数（如 dropout 的 mask）。
   - 打印 `list(model.state_dict().keys())` 看看有哪些参数。

2. **保存 state_dict**  
   - `torch.save(model.state_dict(), "diy.pth")`，把权重存到文件。
   - 可选：用 `torch.save(model.state_dict(), "diy.pth", _use_new_zipfile_serialization=False)` 兼容旧版本 PyTorch。

3. **加载 state_dict**  
   - 先构造一个**相同结构**的模型：`model = DIYForCausalLM(config)`。
   - 加载：`model.load_state_dict(torch.load("diy.pth"))`。
   - 若参数名或 shape 不匹配，会报错；若只想加载匹配的部分，用 `strict=False`：`model.load_state_dict(torch.load("diy.pth"), strict=False)`。

4. **验证加载是否正确**  
   - 用第二步的过拟合实验：训练几步 → 保存 → 加载 → 再训几步。
   - 观察 loss 是否连续（加载后的 loss 应和保存前一致，再训几步应继续下降）。
   - 可选：对比保存前后的参数值（如 `model.model.embed_tokens.weight[0, 0]`），应一致。

5. **自检**  
   - 保存 → 加载 → forward 一次，结果应和保存前一致（相同 seed）。
   - 保存 → 加载 → 再训几步，loss 应连续下降。

**可参考的代码片段（只作提示）**：

```python
# 训练几步
for step in range(10):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(...)
    loss.backward()
    optimizer.step()
    print(f"step {step}, loss = {loss.item():.4f}")

# 保存
torch.save(model.state_dict(), "diy.pth")
print("Saved!")

# 加载
model2 = DIYForCausalLM(config)
model2.load_state_dict(torch.load("diy.pth"))
print("Loaded!")

# 验证：再训几步，loss 应连续
for step in range(10, 15):
    optimizer.zero_grad()
    logits = model2(input_ids)
    loss = F.cross_entropy(...)
    loss.backward()
    optimizer.step()
    print(f"step {step}, loss = {loss.item():.4f}")
```

---

## 4.3 把 config 一起存进 checkpoint

**目的**：除了权重，把配置也存起来，这样加载时不需要手动记住配置，且换配置时不会误用旧权重。

**建议顺序**：

1. **保存时把 config 和 state_dict 一起存**  
   - 保存：`checkpoint = {"config": config, "state_dict": model.state_dict()}`，然后 `torch.save(checkpoint, "diy_checkpoint.pth")`。
   - 这样 checkpoint 文件里既有结构信息（config），也有权重（state_dict）。

2. **加载时先恢复 config，再构造模型，再加载权重**  
   - 加载：`checkpoint = torch.load("diy_checkpoint.pth")`。
   - 取 config：`config = checkpoint["config"]`（如果是 dataclass，可能需要用 `DIYConfig(**checkpoint["config"])` 或直接存 dataclass 对象）。
   - 构造模型：`model = DIYForCausalLM(config)`。
   - 加载权重：`model.load_state_dict(checkpoint["state_dict"])`。

3. **处理 dataclass 的序列化**  
   - 如果 `DIYConfig` 是 dataclass，`torch.save(config, ...)` 可能不能直接序列化。
   - **方式 A**：存成字典：`{"config": config.__dict__, "state_dict": ...}`，加载时 `config = DIYConfig(**checkpoint["config"])`。
   - **方式 B**：用 `dataclasses.asdict(config)` 转成字典，加载时再转回来。
   - **方式 C**：直接用 `torch.save(config, ...)` 试试，新版本 PyTorch 可能支持。

4. **验证完整流程**  
   - 保存：训练几步 → 保存 checkpoint（含 config 和 state_dict）。
   - 加载：加载 checkpoint → 用 config 构造模型 → 加载权重 → 再训几步。
   - 确认 loss 连续，且换 config（如改 `num_layers`）时不会误用旧权重（会报 shape 不匹配）。

5. **自检**  
   - checkpoint 文件包含 config 和 state_dict。
   - 加载后模型结构正确（用 `config` 构造），权重正确（loss 连续）。
   - 换 config 后加载会报错（shape 不匹配），说明不会误用。

**可参考的代码片段（只作提示）**：

```python
# 保存
checkpoint = {
    "config": config.__dict__,  # 或 asdict(config) 或直接 config
    "state_dict": model.state_dict()
}
torch.save(checkpoint, "diy_checkpoint.pth")

# 加载
checkpoint = torch.load("diy_checkpoint.pth")
config = DIYConfig(**checkpoint["config"])  # 如果存的是 dict
model = DIYForCausalLM(config)
model.load_state_dict(checkpoint["state_dict"])
```

---

## 建议时间与自测清单

- **4.1**：约 20～30 分钟。自测：用 `DIYConfig` 构造模型，能正常 forward/backward，改 config 参数后模型结构变化。
- **4.2**：约 15～20 分钟。自测：保存 → 加载 → 再训几步，loss 连续；理解 `strict=False` 的作用。
- **4.3**：约 20～30 分钟。自测：checkpoint 包含 config 和 state_dict，加载后模型结构和权重都正确，换 config 后加载会报错。

---

## 第四步汇总：完成 4.1 + 4.2 + 4.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. Config 类**

- 有一个 **DIYConfig**（dataclass 或普通类），包含所有超参：`vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout`。
- `DIYModel` 和 `DIYForCausalLM` 的 `__init__` 都接收 `config: DIYConfig`，从 `config.xxx` 读参数。
- 改一处 config 即可复现同一模型结构。

**2. state_dict 保存/加载**

- 能保存：`torch.save(model.state_dict(), "diy.pth")`。
- 能加载：先构造相同结构的模型，再 `model.load_state_dict(torch.load("diy.pth"))`。
- 理解 `strict=True/False` 的含义：`True` 要求所有参数都匹配，`False` 只加载匹配的部分。
- 验证：保存 → 加载 → 再训几步，loss 连续。

**3. checkpoint 包含 config**

- 保存：`{"config": config, "state_dict": model.state_dict()}` → `torch.save(checkpoint, "diy_checkpoint.pth")`。
- 加载：先取 config，再 `Model(config)`，再 `load_state_dict`。
- 这样换层数、hidden_size 时不会误用旧权重（会报 shape 不匹配）。
- 验证：完整流程能跑通，且换 config 后加载会报错。

**4. 下一步**

- 有了「能保存/加载、能复现」的能力，就可以进入 **第五步**：完善接口与细节（attention_mask、rms_norm_eps、权重共享、文档），让模型更完整、更易用。

若某一步卡住，优先检查：  
- Config 的参数是否与模型构造时用的参数一致；  
- 保存和加载时的模型结构是否一致（用相同 config 构造）；  
- dataclass 的序列化方式是否正确（存 dict 还是直接存对象）。
