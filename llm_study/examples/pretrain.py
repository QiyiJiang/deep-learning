"""预训练示例：流式 PretrainDataset + scheduler + 定期保存。"""
import argparse
import time
import torch
import math
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

import llm_study
from llm_study import DIYConfig, DIYForCausalLM, PretrainDataset, get_logger

TOKENIZER_DIR = Path(llm_study.__file__).resolve().parent


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

def main():
    logger = get_logger("llm_study.pretrain")
    config = DIYConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=config.default_checkpoints_dir)
    parser.add_argument("--data_path", type=str, default=config.default_pretrain_data_path)
    parser.add_argument("--save_step", type=int, default=config.default_save_step)
    parser.add_argument("--epochs", type=int, default=config.default_num_epochs, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=config.default_lr, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=config.default_accumulation_steps, help="梯度累积步数")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    args = parser.parse_args()

    device = config.device
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    learning_rate = args.learning_rate
    logger.info(f"Using device: {device}")

    # 根据数据行数计算总步数（jsonl 每行一条样本）
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        num_samples = sum(1 for _ in f)
    num_training_steps = max(1, (num_samples // batch_size) * args.epochs)
    logger.info(f"样本数 {num_samples} | batch_size {batch_size} | epochs {args.epochs} | 总步数 {num_training_steps}")

    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler() if use_amp else None
    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    logger.debug(f"pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"pad_token: {tokenizer.pad_token}")
    dataset = PretrainDataset(args.data_path, tokenizer=tokenizer, max_length=config.train_max_length)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    model.train()

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        step = 0
        epoch_start_time = time.time()  # 记录 epoch 开始时间
        
        # Debug: 记录最近几个样本的 hash，检查数据重复
        recent_samples = []
        max_recent = 10

        for batch_ids, batch in enumerate(loader):
            input_ids, labels, loss_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss_mask = loss_mask.to(device)

            current_step = epoch * (num_samples // batch_size) + batch_ids + 1
            lr = get_lr(current_step, num_training_steps, learning_rate)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if use_amp:
                with autocast():
                    logits = model(input_ids)
                    logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
                    labels_slice = labels[:, 1:].reshape(-1)
                    loss = F.cross_entropy(logits_slice, labels_slice, ignore_index=tokenizer.pad_token_id)

                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids)
                logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
                labels_slice = labels[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits_slice, labels_slice, ignore_index=tokenizer.pad_token_id)

                loss = loss / accumulation_steps
                loss.backward()

            if (batch_ids + 1) % accumulation_steps == 0:
                # 计算梯度范数（在 optimizer.step() 之前，clip_grad_norm 之后）
                total_grad_norm = 0.0
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 在 step 之前计算梯度范数
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 在 step 之前计算梯度范数
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                step += 1  # 只在真正更新时增加 step


                if step % args.save_step == 0:
                    checkpoint = {
                        "config": config.__dict__,
                        "state_dict": model.state_dict(),
                    }
                    checkpoint_path = f"{args.checkpoints_dir}/diy_checkpoint.pth"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                # Debug: 记录样本 hash（检查数据重复）
                import hashlib
                sample_hash = hashlib.md5(input_ids.cpu().numpy().tobytes()).hexdigest()[:8]
                recent_samples.append(sample_hash)
                if len(recent_samples) > max_recent:
                    recent_samples.pop(0)
                
                if step % 2000 == 0 or step == 1:
                    # Debug: 检查实际参与 loss 计算的 token 数量
                    labels_slice = labels[:, 1:].reshape(-1)
                    valid_tokens = (labels_slice != tokenizer.pad_token_id).sum().item()
                    total_tokens = labels_slice.numel()
                    
                    # Debug: 检查 logits 的数值范围
                    logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
                    logits_max = logits_slice.max().item()
                    logits_min = logits_slice.min().item()
                    logits_mean = logits_slice.mean().item()
                    
                    # Debug: 检查预测的准确性
                    pred_tokens = logits_slice.argmax(dim=-1)
                    correct = (pred_tokens == labels_slice).float()
                    # 只统计非 padding 位置的准确率
                    valid_correct = correct[labels_slice != tokenizer.pad_token_id].mean().item() if valid_tokens > 0 else 0.0
                    
                    # Debug: 检查数据样本（前几个 token）
                    sample_tokens = input_ids[0, :20].cpu().tolist()
                    sample_text = tokenizer.decode(sample_tokens, skip_special_tokens=False)
                    
                    # Debug: 检查数据重复
                    unique_samples = len(set(recent_samples))
                    duplicate_rate = 1.0 - (unique_samples / len(recent_samples)) if recent_samples else 0.0
                    current_loss = loss.item() * accumulation_steps  # 恢复真实 loss
                    batches_per_epoch = num_samples // batch_size
                    current_batch_in_epoch = batch_ids + 1
                    
                    # 计算时间预估
                    spend_time = time.time() - epoch_start_time
                    eta_min = spend_time / current_batch_in_epoch * batches_per_epoch // 60 - spend_time // 60
                    
                    logger.info(f"step {step} (epoch {epoch+1}, batch {current_batch_in_epoch}/{batches_per_epoch}) | loss = {current_loss:.6f} | lr = {lr:.2e} | epoch_Time: {eta_min}min")
                    logger.debug(f"  loss_mask: {loss_mask.sum().item()}/{loss_mask.numel()} (有效/总数)")
                    logger.debug(f"  实际参与loss的token: {valid_tokens}/{total_tokens} (有效/总数)")
                    logger.debug(f"  logits范围: min={logits_min:.2f}, max={logits_max:.2f}, mean={logits_mean:.2f}")
                    logger.debug(f"  预测准确率(非padding): {valid_correct*100:.2f}%")
                    logger.debug(f"  梯度范数: {total_grad_norm:.4f}")
                    logger.debug(f"  最近{len(recent_samples)}个样本重复率: {duplicate_rate*100:.1f}% (唯一: {unique_samples}/{len(recent_samples)})")
                    logger.debug(f"  样本hash: {sample_hash} | 样本前20个token: {sample_text[:80]}...")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, f"{args.checkpoints_dir}/diy_checkpoint.pth")
    logger.info(f"训练完成，最终 checkpoint 已保存到 {args.checkpoints_dir}/diy_checkpoint.pth")


if __name__ == "__main__":
    main()
