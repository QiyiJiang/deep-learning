"""SFT 示例：SimpleSFTDataset 训练并保存 checkpoint。"""
import argparse
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

import llm_study
from llm_study import DIYConfig, DIYForCausalLM, SimpleSFTDataset, get_logger

TOKENIZER_DIR = Path(llm_study.__file__).resolve().parent


def main():
    logger = get_logger("llm_study.sft")
    config = DIYConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=config.default_sft_data_path)
    parser.add_argument("--checkpoints_dir", type=str, default=config.default_checkpoints_dir)
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=config.default_lr, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=config.default_accumulation_steps, help="梯度累积步数")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--log_interval", type=int, default=10, help="每 N 个 batch 打印一次进度")
    parser.add_argument("--save_step", type=int, default=1000, help="每 N 个 step 保存一次 checkpoint")
    parser.add_argument("--from_checkpoint", type=str, default=f"{config.default_checkpoints_dir}/diy_checkpoint.pth", help="从预训练 checkpoint 加载模型（必需，默认指向预训练结果）")
    args = parser.parse_args()
    device = config.device
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    learning_rate = args.learning_rate
    logger.info(f"Using device: {device}")

    # 检查数据文件
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler() if use_amp else None
    torch.manual_seed(42)
    
    checkpoint_path = Path(args.from_checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"预训练 checkpoint 不存在: {checkpoint_path}")
        logger.error(f"SFT 训练必须从预训练结果加载模型，请先运行预训练或指定正确的 checkpoint 路径")
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {checkpoint_path}")
    
    logger.info(f"从预训练 checkpoint 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型并加载权重
    model = DIYForCausalLM(config)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    logger.info(f"模型加载完成")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params / 1e6:.2f}M | 可训练参数: {trainable_params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    dataset = SimpleSFTDataset(args.data_path, tokenizer, max_length=config.train_max_length)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    
    num_samples = len(dataset)
    batches_per_epoch = len(loader)
    num_training_steps = batches_per_epoch * args.epochs
    logger.info(f"样本数 {num_samples} | batch_size {batch_size} | epochs {args.epochs} | 每轮 {batches_per_epoch} batches | 总步数 {num_training_steps}")

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    model.train()

    step = 0
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()

        for batch_ids, batch in enumerate(loader):
            input_ids, labels, loss_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss_mask = loss_mask.to(device)

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
                    checkpoint_path = f"{args.checkpoints_dir}/diy_checkpoint_sft_step{step}.pth"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

            # 每 log_interval 个 batch 打印一次进度，或每个 epoch 的第一个 batch
            current_batch_in_epoch = batch_ids + 1
            if current_batch_in_epoch % args.log_interval == 0 or current_batch_in_epoch == 1:
                current_loss = loss.item() * accumulation_steps
                spend_time = time.time() - epoch_start_time
                eta_min = spend_time / current_batch_in_epoch * batches_per_epoch // 60 - spend_time // 60
                logger.info(f"step {step} (epoch {epoch+1}, batch {current_batch_in_epoch}/{batches_per_epoch}) | loss = {current_loss:.6f} | lr = {learning_rate:.2e} | epoch_Time: {eta_min}min")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    checkpoint_path = f"{args.checkpoints_dir}/diy_checkpoint_sft_final.pth"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"训练完成，最终 checkpoint 已保存到 {checkpoint_path}")


if __name__ == "__main__":
    main()
