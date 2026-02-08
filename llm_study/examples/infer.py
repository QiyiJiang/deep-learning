"""推理示例：无 cache 与有 KV cache 的生成对比。"""
import torch
from llm_study import DIYConfig, DIYForCausalLM


def main():
    config = DIYConfig()
    device = config.device
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    model = model.to(device)
    model.eval()
    prompt = torch.tensor([[1, 42, 7, 0, 15]], dtype=torch.long, device=device)
    num_generate = 10

    print("=== 无 cache 生成（每次整段 forward）===")
    generated = prompt.clone()
    for step in range(num_generate):
        logits = model(generated)
        next_logits = logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        print(f"step {step}, generated token: {next_token.item()}")

    print("\n=== 有 cache 生成（增量解码）===")
    generated = prompt.clone()
    logits, presents = model(prompt, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    print(f"step 0, generated token: {next_token.item()}")

    for step in range(num_generate - 1):
        new_token = next_token.unsqueeze(1)
        logits, presents = model(new_token, past_key_values=presents, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        print(f"step {step + 1}, generated token: {next_token.item()}")


if __name__ == "__main__":
    main()
