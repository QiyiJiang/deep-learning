# 因果注意力 Mask 错误原因与修正对比（具体例子）

## 1. 约定：谁可以看谁

- **Query 位置 i**：表示「当前在预测第 i+1 个 token」。
- **Key 位置 j**：表示「可以使用的上下文」。
- **因果约束**：位置 i 只能看位置 **0, 1, …, i**，不能看 **i+1, i+2, …**（不能看未来）。

用矩阵表示：`attn[i,j] = True` 表示「query i 可以 attend to key j」。

|       | key 0 | key 1 | key 2 |
|-------|-------|-------|-------|
| q 0   | ✓     | ✗     | ✗     |
| q 1   | ✓     | ✓     | ✗     |
| q 2   | ✓     | ✓     | ✓     |

所以：**下三角（含对角线）为「可见」**，上三角为「不可见」。

---

## 2. PyTorch SDPA 的 bool mask 语义

在 `F.scaled_dot_product_attention(..., attn_mask=..., is_causal=...)` 里：

- **bool attn_mask**：`True` = **可以 attend**，`False` = 被 mask 掉（不能 attend）。
- 也就是说：你要「禁止看」的位置，应该置为 `False`；要「允许看」的位置，置为 `True`。

---

## 3. 原来的错误写法（原因）

原代码（错误）：

```python
attn_mask = torch.triu(torch.ones(seq_len, total_len, ...), diagonal=1)
attn_mask = attn_mask[None, None, :, :]   # [1,1,L,L]
```

`torch.triu(ones, diagonal=1)` 得到的是：**上三角（不含对角线）为 1（True），下三角+对角线为 0（False）**。

以 L=3 为例，矩阵是：

|       | key 0 | key 1 | key 2 |
|-------|-------|-------|-------|
| q 0   | **0** | **1** | **1** |
| q 1   | **0** | **0** | **1** |
| q 2   | **0** | **0** | **0** |

也就是：**「未来」位置被标成 True（可见），「过去+当前」被标成 False（不可见）**，和因果约束正好相反。

结果：

- 位置 0 的 query 可以看 key 1、key 2（看到了未来 token）；
- 位置 1 的 query 可以看 key 2（看到了未来）；
- 模型在算「下一个 token」时已经知道答案，loss 会飞快掉到接近 0，属于**信息泄露**。

---

## 4. 修正思路（两部分的配合）

### 4.1 因果：交给 `is_causal=True`

不再手写 causal 的 bool 矩阵，而是：

```python
att = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.training, ...)
```

训练时 `is_causal=True`，PyTorch 内部保证「query i 只能看 key 0..i」，等价于上面那张**下三角可见**的表，不会再看未来。

### 4.2 Padding：只做「哪些 key 是有效位置」

Padding 的语义是：**无效的 key 位置（padding）不应该被任何 query 看到**。

- 若没有 padding（或暂时不处理），可以 `attn_mask = None`，只靠 `is_causal=True` 即可。
- 若有 `seq_lengths`，需要额外把「超出有效长度的 key」标成不可见。

修正后的写法（只做 padding 的「可见性」）：

```python
if seq_lengths is not None:
    key_pos = torch.arange(total_len, device=x.device)[None, :]   # [1, L]
    valid_k = key_pos < seq_lengths[:, None]                      # [B, L]，True=有效 key
    attn_mask = valid_k[:, None, None, :]                         # [B, 1, 1, L]
else:
    attn_mask = None
```

这里 `attn_mask` 的 shape 是 `[B, 1, 1, L]`，表示「对每个 batch、每个 query 位置，哪些 **key 位置** 是可见的」：  
有效 key 位置为 `True`，padding 的 key 位置为 `False`。  
因果约束已经由 `is_causal=True` 负责，所以不需要在 `attn_mask` 里再画下三角。

---

## 5. 对比小结

| 项目           | 原写法（错误）                         | 修正后                                       |
|----------------|----------------------------------------|----------------------------------------------|
| 因果           | 手写 `triu(..., diagonal=1)` → 上三角 True | 不手写；用 **`is_causal=True`** 由 SDPA 保证 |
| 因果语义       | 未来=True → 能看未来，泄露             | 下三角可见，不能看未来                       |
| Padding        | 再 `attn_mask \| padding_mask` 混在一起 | 单独做「key 是否有效」的 mask，True=可见     |
| 结果           | loss 异常快速掉到 ~0                   | loss 正常缓慢下降                             |

---

## 6. 用 3 个位置再对一下「谁可以看谁」

- **修正后**（`is_causal=True` + 上面的 padding mask）：
  - q0 只能看 k0；
  - q1 只能看 k0, k1；
  - q2 只能看 k0, k1, k2。  
  若某 key 是 padding，再被 `attn_mask` 标成 False，就不会被看到。

- **原写法**（上三角 True）：
  - q0 能看 k1, k2（错误）；
  - q1 能看 k2（错误）；
  - q2 谁也不看（错误）。  
  等价于「只能看未来、不能看过去」，所以 next-token 预测被泄题，loss 猛降。

按上面「按块替换」改完后，行为就是「因果 + padding」都正确；用具体例子理解时，只要记住：**因果 = 下三角可见，SDPA 的 bool mask 里 True = 可见**，就不会再搞反。
