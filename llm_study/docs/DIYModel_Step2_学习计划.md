# 第二步详细学习计划：最小训练循环——验证「能学」

**目标**：用第一步得到的 `DIYForCausalLM` 写一个极简训练循环，对一小段「玩具数据」反复训练，观察 loss 能否明显下降甚至过拟合，从而确认整条链路（前向、loss、反传、优化）正确。  
**前提**：已完成第一步，有可用的 `DIYForCausalLM`，且能根据 logits 和 labels 算 loss、backward。  
**建议**：新建一个单独脚本（例如 `step2_train_loop.py`），或沿用第一步的脚本在下面扩展。

---

## 2.1 准备玩具数据

**目的**：理解「训练一批」需要什么：固定的 `input_ids` 和对应的 `labels`（next-token），以及如何做成可重复喂给模型的小 batch。

**建议顺序**：

1. **定一段固定序列**  
   - 长度不必长，例如 20 个 token。  
   - 内容随意：可以手写一个 list 的 token id（如 `[1, 42, 7, 0, 15, ...]`），注意不要超出词表范围（0 ～ vocab_size-1）。  
   - 用 `torch.tensor([...], dtype=torch.long)` 得到一维的 `seq`，shape 为 `(20,)`。

2. **定 batch 的组成**  
   - **方式 A**：同一段序列复制多份当作一个 batch。例如 `batch_size = 4`，则 `input_ids = seq.unsqueeze(0).expand(4, -1)`，shape 为 `(4, 20)`。这样每个 batch 样本都一样，模型要学的就是「背住」这段序列。  
   - **方式 B**：多段不同序列拼成 batch（每段长度可先做成一样，方便简单实现）。  
   - 建议先做 **方式 A**，过拟合最容易观察。

3. **labels 与 input_ids 的对应**  
   - 若用 next-token 预测：`labels = input_ids[:, 1:]`，即每个位置预测「下一个 token」；算 loss 时用 `logits[:, :-1, :]` 与 `labels` 对齐。  
   - 若当前没有 padding，可先不设 `ignore_index`；若有 padding，在 labels 里把 padding 位置设成 padding_id，并在 `F.cross_entropy(..., ignore_index=padding_id)` 里传入。

4. **自检**  
   - `input_ids.shape` = `(batch_size, seq_len)`，`labels.shape` = `(batch_size, seq_len - 1)`（next-token 时）。  
   - 同一批数据可以多次喂给模型，用于下面的循环。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
vocab_size = 6400  # 与模型一致
seq_len = 20
batch_size = 4
# 固定一段序列（示例：随机但固定 seed）
torch.manual_seed(123)
seq = torch.randint(0, vocab_size, (seq_len,))
input_ids = seq.unsqueeze(0).expand(batch_size, -1)  # (4, 20)
# labels 用于 next-token：在训练时用 logits[:, :-1] 与 input_ids[:, 1:] 算 loss
```

---

## 2.2 写极简训练循环

**目的**：掌握标准的一步训练顺序：`zero_grad` → 前向 → 算 loss → `backward` → `step`，并理解每步的作用。

**建议顺序**：

1. **构造模型和优化器**  
   - 用与第一步一致的超参实例化 `DIYForCausalLM`。  
   - `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)`（或 1e-3，先用小一点更稳）。  
   - 训练前 `model.train()`，保证 dropout 等行为是训练模式。

2. **单步顺序（务必按这个顺序）**  
   - **先** `optimizer.zero_grad()`：把上一步的梯度清掉，避免累积。  
   - **再** 前向：`logits = model(input_ids)`（若你的 `forward` 需要 `is_training=True` 或别的参数，按实际接口传）。  
   - **再** 算 loss：`loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), input_ids[:, 1:].reshape(-1), ...)`，注意 shape 和 next-token 对齐。  
   - **再** `loss.backward()`：反传得到梯度。  
   - **最后** `optimizer.step()`：用当前梯度更新参数。  
   若你的 `DIYForCausalLM.forward` 里**已经**做了 loss 和 backward，则循环里只需：`zero_grad()` → `logits = model(input_ids)`（内部会 backward）→ `optimizer.step()`，并在需要时自己再算一次 loss 用于打印。

3. **用循环包起来**  
   - 例如 `for step in range(100):`，每步按上面顺序执行，并每隔若干步打印 `loss.item()`。  
   - 建议前几步每步都打，确认 loss 是合理数值（正数、非 NaN），再改成每 10 步或 20 步打一次。

4. **自检**  
   - 跑 3～5 步，loss 应是一个正数且无 NaN。  
   - 若 loss 不下降甚至爆炸，可先把 lr 调小（如 1e-4）、或检查 logits/labels 的 shape 和 padding 是否一致。

**可参考的代码片段（只作提示）**：

```python
model = DIYForCausalLM(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

for step in range(100):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")
```

若你的 `forward` 内部已经 `loss.backward()`，则上面改成不在这里算 loss 和 backward，只 `zero_grad()` → `model(input_ids)` → `optimizer.step()`，需要打印时再按同样方式算一次 loss。

---

## 2.3 过拟合这段数据

**目的**：用「能否过拟合」验证模型和优化器没问题：同一小批数据反复训，loss 应明显下降，甚至接近 0。

**建议顺序**：

1. **加大步数**  
   - 在 2.2 的基础上，把步数提高到几百步（如 200～500），或直到 loss 明显稳定在很低的值。  
   - 玩具数据只有一小段且 batch 内重复时，通常几十到一两百步就能看到明显下降。

2. **观察 loss 曲线**  
   - 打印或记录每步（或每 N 步）的 loss。  
   - 期望：先较快下降，然后慢慢变平，最后在一个较低值附近波动。若 loss 能到 0.x 甚至更小，说明模型已经能几乎「背下」这段序列。  
   - 若 loss 不降、或先降后升、或出现 NaN，回头检查：lr 是否过大、forward/backward 是否与 2.2 顺序一致、logits/labels 是否对齐。

3. **可选：看预测是否对齐**  
   - 取最后几步的 `logits`，对 `logits[:, :-1]` 做 `argmax(dim=-1)`，和 `input_ids[:, 1:]` 逐位置比较，看有多少位置预测正确。过拟合好后，正确率应接近 100%。

4. **自检**  
   - loss 在几十到几百步内明显下降。  
   - 同一份数据、同一套超参，多次运行（固定 seed）结果应可复现。

**可参考的代码片段（只作提示）**：

```python
# 在 2.2 的循环基础上，例如跑 300 步
for step in range(300):
    ...
# 最后可打印或画 loss 列表，看曲线
# 可选：pred = logits[:, :-1, :].argmax(dim=-1); acc = (pred == input_ids[:, 1:]).float().mean().item()
```

---

## 建议时间与自测清单

- **2.1**：约 10～15 分钟。自测：`input_ids`、labels（或取法）的 shape 正确，可多次复用同一 batch。
- **2.2**：约 15～20 分钟。自测：单步无报错、loss 为正数且非 NaN，循环几步步数正确。
- **2.3**：约 10～15 分钟。自测：几十到几百步内 loss 明显下降，过拟合小数据时 loss 可到较低值。

---

## 第二步汇总：完成 2.1 + 2.2 + 2.3 后的结果

做完上面三小节后，你应该得到下面这些**统一结果**（方便自检是否达标）：

**1. 数据**

- 有一份固定的 **玩具数据**：`input_ids`（如 shape `(batch_size, seq_len)`）和与之对应的 **labels**（next-token 时即 `input_ids[:, 1:]`），可重复喂给模型。

**2. 训练循环**

- 一个 **极简训练循环**：`zero_grad` → `logits = model(input_ids)` → 用 logits 与 labels 算 **loss** → `loss.backward()` → `optimizer.step()`（若 forward 内已 backward，则循环里只保留 zero_grad、forward、step）。
- 理解并遵守 **顺序**：先清梯度、再前向、再反传、再更新。

**3. 过拟合验证**

- 对同一小批数据反复训练后，**loss 能明显下降**，甚至接近 0（或稳定在很低值）。
- 说明：前向、loss 计算、反传、优化器都正确，模型具备「能学」的能力。

**4. 下一步**

- 有了「能学」的验证，就可以进入 **第三步**：自回归生成（先无 cache、再用 use_cache 做增量解码），以及用 `self.training` 统一训练/推理分支。

若某一步卡住，优先检查：  
- 训练循环顺序是否严格为 zero_grad → forward → loss → backward → step；  
- logits 与 labels 的 shape 是否与 `F.cross_entropy` 要求一致（N×V 与 N）；  
- 若 forward 内已做 backward，循环里是否不再重复 backward，且每步都 zero_grad。
