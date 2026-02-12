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
from llm_study import DIYConfig, DIYForCausalLM, SimplePretrainDataset, get_logger

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
    parser.add_argument("--log_interval", type=int, default=100, help="每 N 个 batch 打印一次进度（如 100 表示 batch 100/44159, 200/44159, ...）")
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
    dataset = SimplePretrainDataset(args.data_path, tokenizer=tokenizer, max_length=config.train_max_length)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    model.train()

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        step = 0
        epoch_start_time = time.time()

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
                    mask_slice = loss_mask[:, 1:].reshape(-1)
                    loss = F.cross_entropy(logits_slice, labels_slice, reduction='none')
                    loss = (loss * mask_slice).sum() / mask_slice.sum().clamp(min=1.0)

                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids)
                logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
                labels_slice = labels[:, 1:].reshape(-1)
                mask_slice = loss_mask[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits_slice, labels_slice, reduction='none')
                loss = (loss * mask_slice).sum() / mask_slice.sum().clamp(min=1.0)

                loss = loss / accumulation_steps
                loss.backward()

            if (batch_ids + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.save_step == 0:
                    checkpoint = {
                        "config": config.__dict__,
                        "state_dict": model.state_dict(),
                    }
                    checkpoint_path = f"{args.checkpoints_dir}/diy_checkpoint.pth"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

            # 每 log_interval 个 batch 打印一次（batch 100/44159, 200/44159, 300/44159, ...）
            current_batch_in_epoch = batch_ids + 1
            if current_batch_in_epoch % args.log_interval == 0:
                current_loss = loss.item() * accumulation_steps
                batches_per_epoch = num_samples // batch_size
                spend_time = time.time() - epoch_start_time
                eta_min = spend_time / current_batch_in_epoch * batches_per_epoch // 60 - spend_time // 60
                logger.info(f"step {step} (epoch {epoch+1}, batch {current_batch_in_epoch}/{batches_per_epoch}) | loss = {current_loss:.6f} | lr = {lr:.2e} | epoch_Time: {eta_min}min")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, f"{args.checkpoints_dir}/diy_checkpoint.pth")
    logger.info(f"训练完成，最终 checkpoint 已保存到 {args.checkpoints_dir}/diy_checkpoint.pth")


if __name__ == "__main__":
    main()
