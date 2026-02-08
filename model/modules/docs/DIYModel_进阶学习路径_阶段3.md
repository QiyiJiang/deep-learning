# 阶段三详细学习计划：训练流程理解

**目标**：理解预训练和 SFT 的区别、各自的作用、以及如何实现。  
**前提**：已完成阶段二（模型规模调整），有可用规模的模型（如 200M–500M）。  
**预计时间**：3-4 天

**建议**：新建单独的练习脚本（例如 `step_pretrain.py`、`step_sft.py`），或参考项目中的 `trainer/train_pretrain.py`、`trainer/train_full_sft.py` 和 `dataset/lm_dataset.py`，不要一开始就改主训练脚本，方便对比和回滚。

---

## 3.1 预训练（Pretraining）原理与实践

**目的**：理解预训练的目标、数据格式、以及如何实现一个简单的预训练脚本。

**建议顺序**：

1. **理解预训练的目标**  
   - **目标**：让模型学习「语言的基本规律」和「世界知识」
   - **任务**：Next-token prediction（给定前文，预测下一个 token）
   - **特点**：
     - 无监督学习（不需要人工标注）
     - 数据量大（TB 级别，实际可先用小数据验证）
     - 训练时间长（大规模时可能需要数周）
     - 学习的是「通用知识」：语法、常识、风格等

2. **理解预训练的数据格式**  
   - **格式**：每行一个 JSON 对象，例如 `{"text": "连续的一段文本..."}`
   - **特点**：文本是**连续的**，没有结构化的问答对，没有「问题」「回答」的划分
   - **处理**：
     - 用 tokenizer 把每段 text 编码成 `input_ids`
     - 按固定长度（如 512）截断或填充，得到多个样本
     - 构造 `labels`：通常 `labels = input_ids`，loss 时用 `logits[:, :-1]` 与 `labels[:, 1:]` 对齐（即预测下一个 token）

3. **理解 loss 的构造**  
   - **Next-token prediction**：  
     - 输入：`input_ids`  shape `(B, L)`  
     - 模型输出：`logits` shape `(B, L, V)`  
     - 预测目标：每个位置预测「下一个 token」，即 `labels = input_ids[:, 1:]`，对应 `logits[:, :-1, :]`  
   - **Loss**：  
     - `loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels.reshape(-1), ignore_index=pad_id)`  
     - 若有 `loss_mask`（如 padding 不参与 loss），则：  
       `loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()`

4. **实现简单的预训练数据加载**  
   - 读取 jsonl，每行 `{"text": "..."}`，收集所有 `text`
   - 对每段 text 用 tokenizer 编码；若长度超过 `max_length` 可截断，不足可 padding 或拼接多段再切分
   - 返回 `(input_ids, labels)` 或 `(input_ids, labels, loss_mask)`，其中 `labels = input_ids`（或右移一位在 loss 里体现）

5. **实现简单的预训练循环**  
   - 设备：`model.to(device)`，数据 `.to(device)`
   - 每个 step：`logits = model(input_ids)`，按上面公式算 loss，`loss.backward()`，`optimizer.step()`
   - 观察：loss 下降即可认为流程正确；生成质量可后续用 infer 简单看

6. **理解预训练的作用与局限**  
   - **作用**：让模型记住大量文本中的模式（语法、常识、风格）
   - **局限**：预训练后的模型**不会对话**——只会「续写」，不会在「该回答」的时候停下来、以对话形式回答

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 预训练数据集（简化版）
class SimplePretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data['text'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)  # (L,)
        # 预训练：labels 与 input_ids 一致，loss 时用 logits[:, :-1] 与 labels[:, 1:]
        labels = input_ids.clone()
        # 可选：padding 位置不参与 loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, loss_mask

# 2. 训练循环（核心）
# for batch in loader:
#     input_ids, labels, loss_mask = batch
#     input_ids = input_ids.to(device)
#     labels = labels.to(device)
#     loss_mask = loss_mask.to(device)
#     logits = model(input_ids)
#     loss_per_token = F.cross_entropy(
#         logits[:, :-1].reshape(-1, vocab_size),
#         labels[:, 1:].reshape(-1),
#         reduction='none'
#     ).view(labels[:, 1:].shape)
#     loss = (loss_per_token * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum().clamp(min=1)
#     loss.backward()
#     optimizer.step()
```

**自检**：
- 能解释预训练的目标（学习语言规律和世界知识）和任务（next-token prediction）。
- 能说明预训练数据格式（连续文本 `{"text": "..."}`）和 labels 的取法（与 input_ids 对齐，预测下一个 token）。
- 能写出最小可运行的预训练循环（含 loss 与 backward），并看到 loss 下降。

**常见问题**：
- **Q: padding 要不要算 loss？**  
  A: 一般不算。用 `loss_mask` 把 padding 位置置 0，只对有效 token 求平均 loss。
- **Q: 数据量很大怎么办？**  
  A: 可以流式读取、按块编码，或先用小文件验证流程，再接入项目里的 `PretrainDataset`。

**学习时间**：约 1-1.5 天

---

### 3.1.1 大规模预训练：数据加载升级

**目的**：在 3.1 小文件预训练跑通的基础上，把数据加载升级为可支撑「大规模、不爆内存」的预训练。现代工程里普遍用**流式 IterableDataset**：不把整份数据读进内存，边读边 yield，配合 DataLoader 做 batch。

**建议顺序**：

1. **理解小文件方案的瓶颈**  
   - `SimplePretrainDataset` 在 `__init__` 里把**整份 jsonl 读进内存**（`samples.append(data['text'])`）。  
   - 数据到 GB 级会 OOM；目标改为：**流式读、内存恒定、可多文件**。

2. **流式 IterableDataset**  
   - 使用 `torch.utils.data.IterableDataset`，不实现 `__len__`，在 `__iter__` 里打开文件、逐行读 jsonl、tokenize 后 yield `(input_ids, labels, loss_mask)`。  
   - 内存里同时只有「当前行」或一个小 buffer，适合 TB 级语料；多文件时在 `__iter__` 里遍历文件列表（可先 shuffle 文件顺序）再逐行读。

3. **实现要点**  
   - **单文件**：`with open(path) as f: for line in f: ... yield ...`。  
   - **多文件**：`data_path` 可为目录或 glob（如 `*.jsonl`），在 `__iter__` 里对文件列表做 `for path in files: ...`，每个文件内再 `for line in f`。  
   - **DataLoader**：`DataLoader(dataset, batch_size=..., num_workers=...)` 会从 `__iter__` 拉取样本并自动 stack 成 batch；如需 shuffle，可在 Dataset 内维护一个小的 buffer（如几千条）做 buffer shuffle 再 yield。  
   - **与项目对接**：在 `dataset/lm_dataset.py` 里新增 `PretrainStreamingDataset`（或替换现有 `PretrainDataset` 的构造方式），训练脚本 `trainer/train_pretrain.py` 里把 Dataset 换成该流式实现即可。

**可参考的代码思路（只作提示，请自己实现或对照项目扩展）**：

```python
import json
from torch.utils.data import IterableDataset

class PretrainStreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_path = file_path  # 也可改为 file_list，支持多文件

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['text']
                enc = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].squeeze(0)
                labels = input_ids.clone()
                loss_mask = (input_ids != self.tokenizer.pad_token_id).long()
                yield input_ids, labels, loss_mask
```

- 使用：`loader = DataLoader(PretrainStreamingDataset(...), batch_size=32, num_workers=0)`（多进程时注意 tokenizer 的线程安全，必要时 `num_workers=0` 或每 worker 单独建 tokenizer）。

**自检**：
- 能说出「全量读入内存」在大规模时的两个问题（内存、扩展性）。
- 能写出一个流式 IterableDataset：`__iter__` 里逐行读 jsonl 并 yield 已 tokenize 的样本，训练脚本用 DataLoader 接入后能正常跑。

**常见问题**：
- **Q: IterableDataset 没有 len，进度条怎么算？**  
  A: 按 step 显示（如每 N step 打印一次），或事先扫一遍得到总行数再传给 Trainer 做总 step 数。
- **Q: num_workers > 0 时 tokenizer 报错？**  
  A: 多进程下避免共享同一 tokenizer；可在每个 worker 的 `__iter__` 里按需创建 tokenizer，或先用 `num_workers=0` 验证流程。

**学习时间**：约 0.5–1 天

---

## 3.2 有监督微调（SFT）原理与实践

**目的**：理解 SFT 的目标、数据格式、以及如何实现一个简单的 SFT 脚本。

**建议顺序**：

1. **理解 SFT 的目标**  
   - **目标**：让模型学习「如何与人对话」或「如何遵循指令」
   - **任务**：Instruction following（给定指令/问题，生成回答）
   - **特点**：
     - 有监督学习（需要高质量的问答/对话数据）
     - 数据量相对预训练小（GB 级别或更少即可起步）
     - 训练时间短（几小时到几天）
     - 学习的是「对话格式」和「指令理解」

2. **理解 SFT 的数据格式**  
   - **格式**：每行一个 JSON，例如  
     `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`  
   - **特点**：结构化的对话，有明确的「问题」和「回答」
   - **处理**：
     - 用 tokenizer 的 `apply_chat_template` 把 conversations 转成一条带特殊 token 的文本
     - 编码成 `input_ids`
     - 构造 `loss_mask`：**只对 assistant 回复部分算 loss**，user 和 system 部分 mask 掉（不参与 loss）

3. **理解为什么只对 assistant 部分算 loss**  
   - 训练目标：让模型「学会在正确的位置生成回答」
   - 若对整段算 loss，模型会学「续写 user 的话」，而不是「生成 assistant 的回复」
   - 因此 loss_mask 在 user/system 段为 0，在 assistant 段为 1（或按 token 粒度构造）

4. **实现简单的 SFT 数据加载**  
   - 读取 jsonl，每行一个 `conversations` 列表
   - 对每条对话用 `apply_chat_template(..., tokenize=False)` 得到字符串，再 tokenize 得到 `input_ids`
   - 根据模板和角色边界，生成 `loss_mask`（assistant 部分为 1，其余为 0）
   - 返回 `(input_ids, labels, loss_mask)`，其中 `labels` 通常就是 `input_ids`（或与 input_ids 一致，用 mask 控制哪些位置参与 loss）

5. **实现简单的 SFT 训练循环**  
   - 与预训练类似：`logits = model(input_ids)`，然后  
     `loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()`  
     其中 `loss_per_token` 由 `logits[:, :-1]` 与 `labels[:, 1:]`（或 `input_ids[:, 1:]`）做 cross_entropy 得到
   - 观察：loss 下降；用 infer 输入一个问题，看是否更倾向于「以回答形式」生成

6. **理解 SFT 的作用与局限**  
   - **作用**：学会对话格式、何时回答、指令理解（如翻译、总结）
   - **局限**：SFT **不能补充知识**——预训练没学过的知识，单靠 SFT 难以补上

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# 1. SFT 数据集（简化版，需根据你的 tokenizer 和 chat_template 调整）
class SimpleSFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data['conversations'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conv = self.samples[idx]
        # 转成 chat 模板字符串（根据 tokenizer 的 chat_template）
        text = self.tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        # 构造 loss_mask：只对 assistant 部分为 1（需根据模板解析角色边界实现）
        loss_mask = self._build_loss_mask(input_ids, conv)
        labels = input_ids.clone()
        return input_ids, labels, loss_mask

    def _build_loss_mask(self, input_ids, conv):
        # 简化：若模板中 assistant 开头有特殊 token，可据此定位并置 1
        # 这里示意：全部为 1 则退化为整段都算 loss（仅用于调试）
        mask = torch.ones_like(input_ids, dtype=torch.long)
        mask[input_ids == self.tokenizer.pad_token_id] = 0
        return mask

# 2. 训练循环与预训练类似，loss 用 loss_mask 加权
# loss = (loss_per_token * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum().clamp(min=1)
```

**自检**：
- 能解释 SFT 的目标（学习对话格式和指令理解）和任务（instruction following）。
- 能说明 SFT 数据格式（conversations + role/content）以及为何只对 assistant 部分算 loss。
- 能写出最小可运行的 SFT 循环（含 loss_mask），并看到 loss 下降。

**常见问题**：
- **Q: loss_mask 怎么精确到「只算 assistant」？**  
  A: 依赖 tokenizer 的 chat_template。可在编码后的序列里根据「assistant 开始/结束」的特殊 token 或位置，把对应区间的 mask 设为 1。可参考项目里 `dataset/lm_dataset.py` 的 SFTDataset 实现。
- **Q: 没有现成 SFT 数据怎么办？**  
  A: 可先用几条手写 conversations 验证流程，再接入公开 SFT 数据（如项目 README 里提到的格式）。

**学习时间**：约 1-1.5 天

---

## 3.3 预训练 vs SFT 的区别与联系

**目的**：深入理解两者的区别、联系，以及为什么需要两步训练。

**建议顺序**：

1. **对比两者的区别**  
   - **数据**：
     - 预训练：大规模无标注连续文本（如 Wikipedia、书籍、网页）
     - SFT：高质量对话/指令数据（问答对、多轮对话）
   - **任务**：
     - 预训练：Next-token prediction（续写）
     - SFT：Instruction following（按指令/问题生成回答）
   - **训练方式**：
     - 预训练：无监督，通常对所有非 padding 位置算 loss
     - SFT：有监督，只对「回答」部分算 loss（loss_mask）
   - **作用**：
     - 预训练：学「知识」与语言规律
     - SFT：学「格式」与何时回答、如何遵循指令

2. **理解两者的联系**  
   - SFT 通常**基于预训练模型**：用预训练权重初始化，再在对话数据上微调
   - 预训练是「打基础」，SFT 是「调格式」
   - 两者缺一不可：
     - 只有预训练 → 只会续写，不会以对话形式回答
     - 只有 SFT（且无预训练）→ 知识不足，回答质量差

3. **对比实验（建议做一次）**  
   - **实验 1**：仅预训练若干 step，不做过 SFT。用同一 prompt 做生成，观察是否倾向于「续写」而不是「回答」。
   - **实验 2**：在实验 1 的 checkpoint 上做 SFT，再同一 prompt 生成，观察是否更像「回答」。
   - **结论**：先预训练再 SFT 的两步流程是必要的。

4. **理解为什么不能一步到位**  
   - **只用 SFT、不做预训练**：  
     - 数据量通常远小于预训练，模型学不到足够语言知识和常识  
     - 容易只会「格式」，内容空洞或错误  
   - **只做预训练、不做 SFT**：  
     - 模型有知识，但不会在「该回答」时以对话形式输出  
     - 用户问一句，模型会续写而不是针对问题作答  

**可参考的对比表（自己整理一遍）**：

| 维度       | 预训练                 | SFT                          |
|------------|------------------------|------------------------------|
| 数据       | 连续文本，无标注       | 对话/指令，有结构            |
| 任务       | Next-token prediction | Instruction following        |
| Loss 范围  | 全序列（除 padding）   | 仅 assistant 部分           |
| 学到的内容 | 知识、语法、风格       | 对话格式、指令理解           |
| 典型数据量 | 很大（TB 级）          | 相对小（GB 级或更少）        |

**自检**：
- 能清晰对比预训练和 SFT 在数据、任务、loss、作用上的区别。
- 能解释为什么需要「先预训练再 SFT」以及「不能只做其中一步」的原因。

**学习时间**：约 1 天

---

## 阶段三汇总：完成 3.1 + 3.2 + 3.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 理解预训练**
- 能解释预训练的目标（学习语言规律和世界知识）和任务（next-token prediction）。
- 理解预训练数据格式（连续文本 `{"text": "..."}`）和 labels 的构造方式。
- 能实现简单的预训练数据加载与训练循环，loss 会下降，模型能「续写」文本。

**2. 理解 SFT**
- 能解释 SFT 的目标（学习对话格式和指令理解）和任务（instruction following）。
- 理解 SFT 数据格式（conversations）以及为何只对 assistant 部分算 loss。
- 能实现简单的 SFT 数据加载与训练循环（含 loss_mask），loss 会下降，模型更会「回答问题」。

**3. 理解两者的区别与联系**
- 能清晰对比预训练和 SFT 的区别（数据、任务、loss、作用）。
- 理解为什么需要两步训练（预训练打基础，SFT 调格式）。
- 能解释为什么不能一步到位（只预训练不会对话，只 SFT 知识不足）。

**4. 实践**
- 至少有一个可跑的预训练脚本和一个可跑的 SFT 脚本（可用小数据、小模型验证）。
- 能根据项目中的 `PretrainDataset` / `SFTDataset` 和 `train_pretrain.py` / `train_full_sft.py` 对照理解并扩展自己的脚本。

**5. 下一步**
- 理解训练流程后，可以进入 **阶段四**：模型优化与工程化（推理优化、Checkpoint、分布式、日志等）。

---

## 学习建议

1. **先理解目标再写代码**：先弄清「预训练学什么」「SFT 学什么」，再实现数据和 loss。
2. **小数据先跑通**：用很少的几条样本先跑通预训练和 SFT 的 loss 与 backward，再考虑大数据和项目里的 Dataset。
3. **对照项目代码**：`dataset/lm_dataset.py` 里的 `PretrainDataset`、`SFTDataset` 以及 `trainer/train_pretrain.py`、`trainer/train_full_sft.py` 可作为对照和扩展参考。
4. **记录笔记**：记下数据格式、loss 写法、遇到的问题和解决办法，方便以后接真实数据与调参。

若某一步卡住，可重点检查：  
- 预训练：`labels` 与 `logits` 的对齐方式（一般是 `logits[:, :-1]` 对 `labels[:, 1:]`），以及 padding 是否被 mask 掉。  
- SFT：`loss_mask` 是否只覆盖 assistant 部分，是否与 tokenizer 的 chat_template 一致。  
- 设备：`model` 与 `input_ids`、`labels`、`loss_mask` 是否都在同一 `device` 上。
