"""SFT 示例：SimpleSFTDataset 训练并保存 checkpoint。"""
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import llm_study
from llm_study import DIYConfig, DIYForCausalLM, SimpleSFTDataset

TOKENIZER_DIR = Path(llm_study.__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/sft_mini_512_15.jsonl")
    args = parser.parse_args()

    config = DIYConfig()
    device = config.device
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    dataset = SimpleSFTDataset(args.data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    model.train()

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
        optimizer.step()
        print(f"loss = {loss.item():.4f}")

    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "diy_checkpoint.pth")


if __name__ == "__main__":
    main()
