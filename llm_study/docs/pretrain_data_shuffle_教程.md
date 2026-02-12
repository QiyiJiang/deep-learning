# 预训练数据 Shuffle 详细教程

你的预训练用的是 **IterableDataset**，按文件行顺序逐条 yield，DataLoader 无法对其做 shuffle。下面分两种做法：**不改 Dataset 类型、在流式里做 buffer shuffle**（推荐），以及**改用 map-style Dataset + DataLoader shuffle**（数据能放进内存时可用）。

---

## 方案一：Buffer Shuffle（推荐，流式 + 打乱）

思路：不一次加载全量数据，而是维护一个**固定大小的缓冲区**；读满后对缓冲区里的样本做一次 shuffle，再逐个 yield，然后继续从文件里填满缓冲区、再 shuffle、再 yield，如此往复。这样既保持流式、不爆内存，又能打乱顺序。

### 1. 参数

- **buffer_size**：缓冲区里存多少条样本再 shuffle 一次。例如 2000 或 5000。
  - 越大：打乱范围越大，更接近「整份数据 shuffle」，但占内存更多、第一次出 batch 更晚。
  - 越小：打乱范围小，但实现简单、内存占用小。
  - 建议：先试 **2000～5000**，数据量特别大再适当加大。

### 2. 在 `PretrainDataset` 里怎么改

在 **`llm_study/datasets.py`** 里：

1. **文件顶部**加一行：`import random`。

2. **`__init__`** 里增加一个参数（例如 `buffer_size: Optional[int] = None`）：
   - 若 `buffer_size` 为 `None` 或 `0`，保持当前行为（不打乱）；
   - 若 `buffer_size > 0`，在 `__iter__` 里用该大小做 buffer shuffle。

3. **`__iter__`** 逻辑改成（保留多 worker 分片）：
   - 先按当前方式按行读文件、按 `worker_id` 分片（只处理 `idx % num_workers == worker_id` 的行）。
   - 若 **未开 buffer**（buffer_size 为 0 或 None）：和现在一样，直接 `yield input_ids, labels, loss_mask`。
   - 若 **开了 buffer**：
     - 建一个列表 `buffer = []`。
     - 每读出一条且通过分片条件的样本，先 **append 进 buffer**。
     - 当 `len(buffer) >= buffer_size` 时：
       - `random.shuffle(buffer)`
       - 逐个 `yield buffer.pop(0)` 或 `for x in buffer: yield x`，然后 `buffer.clear()`。
     - 文件读完后，若 buffer 里还有剩余，再 shuffle 一次，再 yield 完。

4. **多 worker 注意**：每个 worker 只看到自己分片到的行，所以每个 worker 内部单独维护一个 buffer、单独 shuffle 即可，不需要跨 worker 同步。

### 3. 训练脚本里怎么传

已在 **`llm_study/examples/pretrain.py`** 里加上参数 **`--buffer_size`**，默认 0（不 shuffle）。开 shuffle 时直接传正整数即可，例如：

```bash
uv run python -m llm_study.examples.pretrain --buffer_size 2000
```

即每读满 2000 条样本做一次 shuffle 再 yield。也可在代码里写死默认值（例如 2000），见 `PretrainDataset(..., buffer_size=args.buffer_size if args.buffer_size > 0 else None)`。

这样 DataLoader 拉取的 batch 顺序就是「先一段打乱、再下一段打乱」，不会整文件固定顺序。

### 4. 可选：每轮 shuffle 更彻底

若希望每个 epoch 的 shuffle 不同，可以在 Dataset 里加一个 `seed` 或 `epoch` 的占位（IterableDataset 一般拿不到 epoch），或在训练脚本里每个 epoch 新建一次 Dataset（并传入不同 `random_seed`），在 `__iter__` 里用 `random.seed(self.random_seed + epoch)`。更简单的是：不传 epoch，仅用 buffer shuffle，每轮文件顺序不变但 buffer 内打乱，通常已经能明显缓解「前一段被记住」的问题。

---

## 方案二：Map-Style Dataset + DataLoader(shuffle=True)

思路：一次性把样本都加载进内存（或可索引的存储），用 **map-style Dataset**（实现 `__len__` 和 `__getitem__`），DataLoader 里设 **`shuffle=True`**，由 PyTorch 每轮打乱索引。

### 1. 何时适用

- 数据量能接受**全量加载**（例如几十万条、且每条已 tokenize 成 tensor 能放进内存）。
- 你项目里已有 **`SimplePretrainDataset`**（map-style），可直接用。

### 2. 用法示例

```python
from llm_study import SimplePretrainDataset

dataset = SimplePretrainDataset(args.data_path, tokenizer=tokenizer, max_length=config.train_max_length)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,   # 每个 epoch 打乱
    num_workers=4,
    pin_memory=True,
)
```

注意：`SimplePretrainDataset` 会在 `__init__` 里读入整个 jsonl 的 `text` 列表，**不会**在 init 里做 tokenize（是在 `__getitem__` 里按需 tokenize），所以内存里主要是字符串；若样本很多、很长，仍可能 OOM，此时用方案一更稳。

---

## 小结

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Buffer shuffle** | 流式、省内存、不改 DataLoader 用法 | 打乱只在 buffer 内，不是全局 |
| **Map-style + shuffle=True** | 每轮全局打乱，和 MiniMind 一致 | 需能承受全量加载/索引 |

推荐：在 **`PretrainDataset`** 里加 **buffer_size**，训练时传 **buffer_size=2000**（或 5000），即可在不大改代码、不爆内存的前提下给数据加 shuffle，减轻「前一段被记住、loss 异常掉」的现象。
