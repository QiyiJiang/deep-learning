"""小规模训练示例：若干 step 训练 + 保存/加载 checkpoint。"""
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

import llm_study
from llm_study import DIYConfig, DIYForCausalLM

TOKENIZER_DIR = Path(llm_study.__file__).resolve().parent


def main():
    config = DIYConfig()
    batch_size, seq_len = 2, 10
    device = config.device
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    model.train()

    for step in range(5):
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
        labels_slice = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits_slice, labels_slice)
        loss.backward()
        optimizer.step()
        print(f"step {step + 1}, loss = {loss.item():.4f}")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "diy_checkpoint.pth")

    checkpoint = torch.load("diy_checkpoint.pth", map_location=device)
    config2 = DIYConfig(**checkpoint["config"])
    model2 = DIYForCausalLM(config2)
    model2 = model2.to(device)
    model2.load_state_dict(checkpoint["state_dict"])

    for step in range(5, 10):
        optimizer.zero_grad()
        logits = model2(input_ids)
        logits_slice = logits[:, :-1, :].reshape(-1, config2.vocab_size)
        labels_slice = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits_slice, labels_slice)
        loss.backward()
        optimizer.step()
        print(f"step {step + 1}, loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
