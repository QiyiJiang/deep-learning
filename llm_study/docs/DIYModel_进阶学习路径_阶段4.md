# 阶段四详细学习计划：模型优化与工程化

**目标**：在「小规模预训练和 SFT 已跑通」的基础上，先做好**训练优化**（稳定、可恢复、可观测），再做**部署与推理优化**，使模型真正可工程化使用。  
**前提**：已完成阶段三（预训练和 SFT），有小规模数据和训练脚本可跑。  
**预计时间**：4–6 天

**建议学习顺序**：先 **4.0 大规模训练前的知识补充**（避免长训翻车），再 **4.1 训练优化**，最后 **4.2 部署与推理优化**。

---

## 4.0 大规模训练前的知识补充（建议先学）

**目的**：你已完成小规模预训练和微调，在正式使用**大规模数据集**之前，先补足「训练可恢复、训练稳定、可观测」等知识，避免长时间训练中途崩溃或无法复现。

**建议顺序**：

1. **Checkpoint 与恢复训练**  
   - **为什么**：大规模训练可能跑几天甚至几周，断点、掉电、OOM 后若不能从某一步恢复，前面的算力就白费。  
   - **要会**：定期保存 `model.state_dict()`、`optimizer.state_dict()`、`step`/`epoch`、以及配置；恢复时加载并从中断步继续。  
   - **建议**：先实现「每 N step 或每 epoch 存一份」，再实现「启动时若发现已有 checkpoint 则自动从该步继续」。

2. **学习率与 warmup**  
   - **为什么**：大 batch、长序列时，一开始用大学习率容易不稳定；warmup 可先小步走再升到目标 lr。  
   - **要会**：使用带 warmup 的 scheduler（如 `get_linear_schedule_with_warmup`），理解 warmup_steps 和 total_steps 的含义。  
   - **建议**：在小数据上对比「有/无 warmup」的 loss 曲线，观察前几百步是否更平滑。

3. **梯度裁剪（Gradient Clipping）**  
   - **为什么**：长序列或异常样本可能导致梯度爆炸，训练发散。  
   - **要会**：在 `loss.backward()` 之后、`optimizer.step()` 之前加 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`。  
   - **建议**：先设一个常用值（如 1.0），观察训练是否更稳定；若仍不稳定可适当减小。

4. **Loss 与日志监控**  
   - **为什么**：长训时需要知道当前 loss、是否 NaN、大致进度，便于判断是否正常、何时停训。  
   - **要会**：每 N step 打印或写日志（loss、lr、step）；可选 tensorboard/wandb 做曲线。  
   - **建议**：至少实现「每 100 step 打印一次 loss」，再考虑接 wandb/tensorboard。

5. **数据量与显存估算（可选）**  
   - **为什么**：上大规模前，心里有数：当前 batch_size × seq_len 下显存是否够、大概能撑多大模型/多长序列。  
   - **要会**：粗算「模型参数量 × 4 字节（fp32）或 × 2（fp16）」+ 激活/优化器状态；或先用小 batch 试跑，再按比例估算。  
   - **建议**：在 4.1 做混合精度或 DDP 前，先单卡跑通并记录「最大不 OOM 的 batch_size」。

**自检**：  
- 能说出「为什么大规模训练前要先做 checkpoint、warmup、梯度裁剪、日志」。  
- 能写出「保存/加载 checkpoint」和「带 warmup 的 scheduler」的最小可用代码。

**学习时间**：约 1 天

---

## 4.1 训练优化

**目的**：让训练**稳定、可恢复、可扩展**（多卡、多 worker、可选混合精度），为大规模数据训练打基础。

**建议顺序**：

1. **Checkpoint 管理（巩固 4.0）**  
   - 保存：`model.state_dict()`、`optimizer.state_dict()`、`step`、`epoch`、`config`（或 lr_scheduler 状态）。  
   - 加载：从文件恢复上述内容，并从 `step`/`epoch` 继续训练。  
   - 策略：每 N step 或每 epoch 存一份，保留最近 K 个（避免占满磁盘）。

2. **分布式训练（DDP）**  
   - **问题**：单卡显存/速度有限，想用多卡加速。  
   - **要点**：用 `DistributedDataParallel` 包装模型，用 `DistributedSampler` 让每张卡读不同数据，用 `torchrun` 启动。  
   - **注意**：每进程需设置 `local_rank`/`rank`，模型与数据放到对应 `cuda:rank`。

3. **日志与监控**  
   - 基础：每 N step 打印 loss、lr、throughput（tokens/s 或 samples/s）。  
   - 可选：接 wandb 或 tensorboard，记录 loss、lr 曲线，便于对比实验。

4. **数据加载（与阶段三 3.1.1 衔接）**  
   - 大规模预训练/SFT 已用流式 IterableDataset + worker 分片（见阶段三文档）。  
   - 这里可巩固：`num_workers`、`pin_memory`、`prefetch_factor` 对吞吐的影响；多卡时 DataLoader 与 DistributedSampler 的配合。

5. **梯度累积（可选）**  
   - **问题**：显存只够小 batch，但希望等效大 batch 训练。  
   - **要点**：每累积 N 个小 batch 再 `optimizer.step()` 并 `zero_grad()`，等效 batch = 单次 batch × N。

6. **混合精度（可选）**  
   - **问题**：想省显存、提速度。  
   - **要点**：用 `torch.cuda.amp.autocast` 和 `GradScaler`，前向与 loss 在 fp16，梯度用 scaler 缩放再反传，避免下溢。

**可参考的代码片段（只作提示，请自己敲一遍）**：

```python
# Checkpoint 保存与加载（示意）
def save_checkpoint(model, optimizer, step, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step, 'epoch': epoch,
    }, path)

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('step', 0), ckpt.get('epoch', 0)

# 梯度裁剪（在 backward 之后、step 之前）
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**自检**：  
- 能实现「定期保存 + 启动时恢复」的完整训练循环。  
- 能说明 DDP 下 DataLoader 与 Sampler 的配合方式；若有多卡，能跑通 DDP 训练。

**学习时间**：约 2 天

---

## 4.2 部署与推理优化

**目的**：训练好的模型要能**快速、省显存地推理**，并便于保存/加载和简单部署。

**建议顺序**：

1. **模型保存与加载**  
   - **训练用**：checkpoint 含 optimizer、step 等（见 4.1）。  
   - **推理用**：通常只保存 `model.state_dict()` 和配置（如 vocab_size、hidden_size 等），推理脚本里先构建相同结构的模型再 `load_state_dict`。  
   - **要会**：区分「训练恢复用 ckpt」和「推理用权重文件」的保存格式与加载方式。

2. **量化（Quantization）**  
   - **问题**：fp32 模型占显存大、推理慢。  
   - **方法**：动态量化（如 `torch.quantization.quantize_dynamic` 对 Linear 转 int8）、或使用业界工具（如 bitsandbytes、GPTQ）。  
   - **效果**：模型体积和显存约降为约 1/4，推理速度通常有 2–3 倍提升；需验证生成质量是否可接受。

3. **KV Cache 优化**  
   - **原理**：自回归生成时，已计算过的 K/V 可缓存，避免重复计算。  
   - **你已实现**：若 DIYModel/Attention 里已有 past_key_values 和 use_cache，即已在用 KV Cache。  
   - **可选**：了解「量化 KV Cache」「PagedAttention」等进一步优化思路。

4. **Batch 推理**  
   - **原理**：一次对多条序列做 forward，提高 GPU 利用率。  
   - **要点**：多条序列需 padding 到同一长度（或使用 dynamic batching），注意 mask 与 padding 位置。

**可参考的代码片段（只作提示）**：

```python
# 动态量化（示意）
import torch
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# 推理时用 quantized_model 替代 model
```

**自检**：  
- 能保存「仅权重+配置」的推理用文件，并在新脚本中加载并生成。  
- 能说明量化的作用（省显存、加速）；若实现量化，能对比量化前后速度或显存。

**学习时间**：约 1–2 天

---

## 阶段四汇总：完成 4.0 + 4.1 + 4.2 后的结果

做完上述内容后，你应该达到：

**1. 大规模训练前（4.0）**  
- 理解并会使用 checkpoint 恢复、warmup、梯度裁剪、基础日志。  
- 在大规模数据正式开训前，训练流程已具备「可恢复、可观测、更稳定」的基础。

**2. 训练优化（4.1）**  
- 能实现完整 checkpoint 管理（保存 + 恢复）。  
- 理解并能在多卡上使用 DDP；能配合日志/监控观察训练。  
- 数据加载与阶段三的流式/分片方案衔接良好；可选掌握梯度累积、混合精度。

**3. 部署与推理优化（4.2）**  
- 能区分训练用 checkpoint 与推理用权重，并正确保存与加载。  
- 理解量化、KV Cache、batch 推理的作用，并能做简单推理优化。

**4. 下一步**  
- 在此基础上可**正式上大规模数据**做预训练/SFT；若需进一步扩展，可再学习更多分布式策略、更大模型与数据规模等。

---

## 学习建议

1. **先 4.0 再 4.1**：把「可恢复、可观测、稳定」做扎实，再上大规模数据和多卡，避免长训翻车。  
2. **训练优化优先于推理优化**：先保证「能稳定训完、能存能恢复」，再考虑部署与推理加速。  
3. **小数据验证**：checkpoint、DDP、混合精度等都可先用小数据或小步数验证流程，再放大。  
4. **与阶段三衔接**：数据加载与 worker 分片已在阶段三 3.1.1 完成，阶段四重点放在训练流程与推理侧即可。
