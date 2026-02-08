"""预训练示例：流式 PretrainDataset + scheduler + 定期保存。"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import llm_study
from llm_study import DIYConfig, DIYForCausalLM, PretrainDataset

TOKENIZER_DIR = Path(llm_study.__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_hq_15.jsonl")
    parser.add_argument("--save_step", type=int, default=2000)
    args = parser.parse_args()

    config = DIYConfig()
    device = config.device
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_training_steps = 1_400_000
    num_warmup_steps = int(num_training_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    dataset = PretrainDataset(args.data_path, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    model.train()
    step = 0

    for batch in loader:
        optimizer.zero_grad()
        input_ids, labels, loss_mask = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)
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
            torch.save(checkpoint, f"diy_checkpoint_step_{step}.pth")

        if step % 200 == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step} | loss = {loss.item():.4f} | lr = {lr:.2e}")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "diy_checkpoint_last.pth")


if __name__ == "__main__":
    main()
