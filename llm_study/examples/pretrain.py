"""预训练示例：流式 PretrainDataset + scheduler + 定期保存。"""
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import llm_study
from llm_study import DIYConfig, DIYForCausalLM, PretrainDataset

TOKENIZER_DIR = Path(llm_study.__file__).resolve().parent


def main():
    config = DIYConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default=config.default_checkpoints_dir)
    parser.add_argument("--data_path", type=str, default=config.default_pretrain_data_path)
    parser.add_argument("--save_step", type=int, default=config.default_save_step)
    parser.add_argument("--epochs", type=int, default=config.default_num_epochs, help="训练轮数")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    args = parser.parse_args()

    device = config.device
    batch_size = config.default_batch_size
    print(f"Using device: {device}")

    # 根据数据行数计算总步数（jsonl 每行一条样本）
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        num_samples = sum(1 for _ in f)
    num_training_steps = max(1, (num_samples // batch_size) * args.epochs)
    num_warmup_steps = int(num_training_steps * 0.05)
    print(f"样本数 {num_samples} | batch_size {batch_size} | epochs {args.epochs} | 总步数 {num_training_steps}")

    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler() if use_amp else None
    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.default_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    dataset = PretrainDataset(args.data_path, tokenizer=tokenizer, max_length=config.train_max_length)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0

    for batch in loader:
        optimizer.zero_grad()
        input_ids, labels, loss_mask = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)

        if use_amp:
            with autocast():
                logits = model(input_ids)
                logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
                labels_slice = labels[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits_slice, labels_slice)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)
            logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
            labels_slice = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits_slice, labels_slice)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        step += 1

        if step % args.save_step == 0:
            checkpoint = {
                "config": config.__dict__,
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, f"{args.checkpoints_dir}/diy_checkpoint_step_{step}.pth")

        if step % 200 == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step}/{num_training_steps} | loss = {loss.item():.4f} | lr = {lr:.2e}")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, f"{args.checkpoints_dir}/diy_checkpoint_last.pth")


if __name__ == "__main__":
    main()
