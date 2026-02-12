# DIY 预训练 loss 异常快速下降原因分析

## 现象对比

| 脚本 | batch 100 | batch 800 | batch 1300 | 趋势 |
|------|-----------|-----------|------------|------|
| **MiniMind** | loss 7.10 | loss 5.75 | loss ~4.4 (batch 1600) | 缓慢、平稳下降 |
| **你的 DIY** | loss 6.96 | loss 2.96 | loss **0.27** | 千 batch 内掉到接近 0 |

DIY 在约 1300 个 batch 内从 ~7 掉到 0.27，属于**异常快速下降**，通常是**过拟合/记忆**当前数据，而不是在学泛化能力。

---

## 主要原因

### 1. Dropout = 0（当前 config 为 0）

你当前的 `llm_study/config.py` 里是 **`dropout: float = 0.0`**。

- 无 dropout 时，模型几乎没有正则，很容易在少量步数内「记住」当前 batch 或连续一段数据，表现为 loss 迅速接近 0。
- 之前分析里建议设为 **0.05～0.1**（如 0.1），你这边若被改回 0，就会重新出现「loss 异常下降」。

**建议**：在 `llm_study/config.py` 中把 `dropout` 改为 **0.1**（或先试 0.05），确认模型里各处都用了 `config.dropout`（embed、Attention、FFN 等已有），然后重跑一段看 loss 是否还会在千 batch 内压到 0.2～0.3。

---

### 2. 数据顺序固定、无 shuffle

- 你的 **PretrainDataset** 是 **IterableDataset**，按文件行顺序 **逐行 yield**，**没有 shuffle**。
- **DataLoader** 对 IterableDataset 不会做 shuffle，所以每个 epoch 的 **batch 顺序完全一致**（文件顺序）。
- 若文件前半段主题/风格集中或重复多，模型会很快拟合这一段，loss 掉得特别快；MiniMind 若用 map-style Dataset + `shuffle=True`，每轮顺序打乱，不容易出现这种「连续记忆一段」的情况。

**建议**：

- **短期**：先把 **dropout 调到 0.1**，观察是否明显缓解。
- **中期**：在 IterableDataset 里做 **buffer shuffle**：例如维护一个 2000～5000 条的 buffer，读满后 shuffle 再 yield，再继续读文件填 buffer（类似「流式 + 局部打乱」），这样同一 epoch 内顺序不再完全固定，减少对「前 N 条」的快速记忆。

---

### 3. 其他因素（次要）

- **序列长度**：DIY 用 512（若未改），MiniMind 用 340；更长序列每 batch 更多 token，梯度更稳，但不会单独导致「掉到 0.27」，主因仍是 dropout 和数据顺序。
- **学习率**：两边都是 5e-4，不是主因。
- **Loss 写法**：已用显式 mask，与 MiniMind 对齐，无异常。

---

## 建议修改顺序

1. **立刻改**：在 `llm_study/config.py` 中设置 **`dropout: float = 0.1`**，重跑预训练，看 loss 是否还会在 1000～2000 batch 内掉到 0.2～0.3。
2. **若仍掉得快**：在 `llm_study/datasets.py` 的 PretrainDataset 的 `__iter__` 里加 **buffer shuffle**（例如 buffer_size=2000，读满 shuffle 再 yield），再观察。
3. **可选**：若希望和 MiniMind 更一致，可把 `train_max_length` 改为 340，主要影响是每步计算量和速度，对「是否异常下降」影响小于 dropout 和 shuffle。

改完 dropout 后，预期现象是：**首步 loss 仍在 8～9 左右，随后缓慢下降，千 batch 内大致在 4～6 区间**，而不是掉到 0.2～0.3。

---

## 附录：MiniMind 的 dropout 与数据顺序

### Dropout

- **MiniMind**：在 `model/model_minimind.py` 的 `MiniMindConfig` 里，**`dropout: float = 0.0`**（默认也是 0）。
- 因此 MiniMind 默认同样是 **dropout=0**；若你看到其 loss 缓慢下降而 DIY 掉得很快，主要差异更可能来自**数据顺序**，而不是 dropout。

### 数据顺序（shuffle）

- **MiniMind**（`trainer/train_pretrain.py`）：
  - 使用 **map-style Dataset**（`dataset.lm_dataset.PretrainDataset`，有 `__len__`），而不是 IterableDataset。
  - DataLoader 构造：  
    `DataLoader(train_ds, batch_size=..., shuffle=(train_sampler is None), sampler=train_sampler, ...)`  
  - **单卡**：`train_sampler is None` → **`shuffle=True`**，每个 epoch 打乱样本顺序。  
  - **多卡**：用 **`DistributedSampler`**，每轮 `train_sampler.set_epoch(epoch)`，各卡拿到不同、且每轮重新打乱的索引。
- **你的 DIY**：
  - 使用 **IterableDataset**（`llm_study/datasets.py` 的 `PretrainDataset`），按文件行顺序 yield。
  - DataLoader 未传 `shuffle`（IterableDataset 本身也不支持 DataLoader 的 shuffle），因此**顺序固定**，等于无 shuffle。

**结论**：MiniMind 通过 **map-style Dataset + shuffle=True（或 DistributedSampler）** 每轮打乱顺序；DIY 是 **IterableDataset + 无 shuffle**，顺序固定。要逼近 MiniMind 行为，除可选地把 dropout 调到 0.1 外，建议在 DIY 的 IterableDataset 里做 **buffer shuffle**，或在能接受全量加载时改用 map-style Dataset + DataLoader(shuffle=True)。
