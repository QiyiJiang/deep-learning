"""统计参数量与显存占用估算。"""
import torch
from llm_study import DIYConfig, DIYForCausalLM


def main():
    config = DIYConfig()
    device = config.device
    print(f"Using device: {device}")

    model = DIYForCausalLM(config)
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    if torch.cuda.is_available():
        memory_fp32 = total * 4 / 1024**3
        memory_fp16 = total * 2 / 1024**3
        print("\n显存占用估算（仅参数）：")
        print(f"  FP32: {memory_fp32:.2f} GB")
        print(f"  FP16: {memory_fp16:.2f} GB")
        print(f"  训练时（FP32 + 梯度 + 优化器）：约 {memory_fp32 * 3:.2f} GB")


if __name__ == "__main__":
    main()
