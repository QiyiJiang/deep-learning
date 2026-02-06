# DIYModel 完善路线（在现有代码之后的下一步）

**已有代码**：`DIYModel`（embed → ModelBlock × N → RMSNorm）、训练/推理双分支、`past_key_values` / `presents` 的 KV cache 接口。这些是你已经学过的内容。

下面是在此基础上**如何一步步完善模型**的 todolist，按顺序做即可由浅入深。

---

## 第一步：从 hidden_states 到 logits——加上语言模型头

**目标**：当前模型只输出「表示」；要能做语言模型训练和生成，需要得到词表上的分数（logits）。

- [ ] **1.1 加 lm_head**  
  在单独脚本里先试：`lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`，对 `DIYModel` 的 `hidden_states` 做一次线性变换得到 `logits`，shape 应为 `(batch, seq_len, vocab_size)`。  
  **学习点**：语言模型最后一层就是把「每个位置的表示」映射到「对词表每个 token 的分数」。

- [ ] **1.2 用 logits 算 loss 并 backward**  
  构造 `labels`（如 next-token：`input_ids[:, 1:]`，logits 取 `[:, :-1]`），用 `F.cross_entropy(..., ignore_index=padding_id)` 算 loss，再 `loss.backward()`。确认能正常反传。  
  **学习点**：next-token prediction 的 label 对齐方式，以及 padding 的 ignore_index。

- [ ] **1.3 封装成「完整语言模型」类**  
  写一个类（如 `DIYForCausalLM`），内部包含 `DIYModel` + `lm_head`，`forward(input_ids, ...)` 返回 `logits`。之后所有训练和生成都基于这个类，不动底层 `DIYModel`。  
  **学习点**：backbone 与 head 的分离，便于以后换 head 或复用 backbone。

---

## 第二步：最小训练循环——验证「能学」

**目标**：确认加上 lm_head 后，整条链路能训练、能收敛。  
**详细学习计划**：见 [DIYModel_Step2_学习计划.md](DIYModel_Step2_学习计划.md)。

- [ ] **2.1 准备玩具数据**  
  固定一小段 token 序列（如 20 个），复制成多份做成小 batch，或直接 `input_ids` + `labels` 的 tensor。  
  **学习点**：输入与 label 的对应（同长度或右移一位）。

- [ ] **2.2 写极简训练循环**  
  `AdamW` + 循环里：`logits = model(input_ids, ...)`，`loss = cross_entropy(logits, labels)`，`loss.backward()`，`optimizer.step()`，`optimizer.zero_grad()`。  
  **学习点**：backward / step / zero_grad 的顺序与作用。

- [ ] **2.3 过拟合这段数据**  
  观察 loss 能否在几十步内明显下降甚至接近 0。能过拟合说明：前向、loss、反传都没问题，可以进入「生成」和「工程化」。  
  **学习点**：过拟合是验证模型与优化器正确性的最小实验。

---

## 第三步：自回归生成与 KV cache 的正确用法

**目标**：会用现有 `past_key_values` / `presents` 做增量解码，并统一用 `model.train()` / `model.eval()` 控制分支。  
**详细学习计划**：见 [DIYModel_Step3_学习计划.md](DIYModel_Step3_学习计划.md)。

- [ ] **3.1 先实现「无 cache」的逐 token 生成**  
  循环：每次只预测下一个 token（取 logits 最后一维 argmax 或采样），拼到序列后，整段再送进 model。理解这样每步都重算整段 attention，复杂度随步数平方增长。  
  **学习点**：自回归生成的本质是「用已生成序列 + 新 token 预测下一个」。

- [ ] **3.2 再用 use_cache 做增量解码**  
  第一次：整段 prompt 进 model，`use_cache=True`，得到 `logits, presents`。之后每步：只送「新生成的 1 个 token」，把上一轮的 `presents` 当作 `past_key_values` 传入，得到新的 `logits, presents`。可选：和 3.1 在相同 prompt + seed 下对比几步的 logits 是否一致。  
  **学习点**：你当前 `(cached, cached_pos)` 的格式在逐步生成时如何传递与更新。

- [ ] **3.3 用 self.training 替代 is_training 参数**  
  在 `DIYModel.forward` 里改为 `is_training = self.training`（或保留参数为可选 override）。调用方只通过 `model.train()` / `model.eval()` 控制，不再传 `is_training`。  
  **学习点**：PyTorch 的 train/eval 模式与 `training` 属性的一致性。

---

## 第四步：配置与保存——方便复现

**目标**：超参集中管理，能保存/加载权重（和配置），便于复现和继续改。

- [ ] **4.1 引入简单 Config**  
  用 dataclass 或普通类存 `vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout` 等。`DIYModel` 和 `DIYForCausalLM` 的 `__init__` 改为接收一个 `config`，从 `config.xxx` 读参数。  
  **学习点**：改一处配置即可复现同一模型结构。

- [ ] **4.2 保存与加载 state_dict**  
  `torch.save(model.state_dict(), "diy.pth")`，加载时 `model.load_state_dict(torch.load("diy.pth"))`。用第二步的过拟合实验：保存 → 加载 → 再训几步，看 loss 是否连续。  
  **学习点**：state_dict 内容、加载时 `strict` 的含义。

- [ ] **4.3 把 config 一起存进 checkpoint**  
  保存：`{"config": config, "state_dict": model.state_dict()}`；加载：先取 config，再 `Model(config)`，再 `load_state_dict`。这样换层数、hidden_size 时不会误用旧权重。  
  **学习点**：checkpoint 里除了权重，结构信息如何保存与恢复。

---

## 第五步：完善接口与细节（按需选做）

**目标**：让模型更完整、更易用、更易维护。

- [ ] **5.1 支持 attention_mask**  
  若希望和「按 mask 算有效长度」的用法一致：在封装层（如 `DIYForCausalLM`）接收 `attention_mask`，从中得到 `seq_lengths = attention_mask.sum(dim=1)`，再传给 `DIYModel`。底层仍用现有 `seq_lengths` 做 padding mask，不改 `ModelBlock` 接口。  
  **学习点**：attention_mask 与 seq_lengths 的等价关系，以及在哪里做转换。

- [ ] **5.2 RMSNorm 的 eps 可配置**  
  在 Config 里加 `rms_norm_eps`，构造 `DIYModel` 时传给 `RMSNorm(hidden_size, eps=config.rms_norm_eps)`。  
  **学习点**：归一化里小常数对数值稳定性的作用。

- [ ] **5.3 权重共享（embed 与 lm_head）**  
  在 `DIYForCausalLM` 里令 `lm_head.weight = embed_tokens.weight`，观察参数量与训练曲线变化。  
  **学习点**：很多 LLM 用 input/output embedding 共享减少参数、对齐表示。

- [ ] **5.4 类型注解与 docstring**  
  给 `DIYModel.forward` 和 `DIYForCausalLM.forward` 补上参数/返回值的类型注解和一两句说明（返回什么、何时带 presents）。  
  **学习点**：让接口一目了然，便于后续扩展。

---

## 顺序小结

| 步骤 | 内容 |
|------|------|
| 一 | lm_head → logits → loss → 封装成 DIYForCausalLM |
| 二 | 玩具数据 + 最小训练循环 + 过拟合验证 |
| 三 | 逐 token 生成（无 cache / 有 cache）+ self.training |
| 四 | Config + state_dict 保存/加载 + config 进 checkpoint |
| 五 | attention_mask、rms_norm_eps、权重共享、文档（选做） |

按 **1 → 2 → 3 → 4 → 5** 做下去，就是在你**已有 backbone 和 cache 接口**之上，把模型完善成「可训练、可生成、可保存、可复现」的完整语言模型，每一步都是对现有代码的增量改进。
