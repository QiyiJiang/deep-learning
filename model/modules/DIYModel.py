from cProfile import label
from math import log
import torch
from torch import nn
import torch.nn.functional as F
from model.modules.RMSNorm import RMSNorm
from model.modules.ModelBlock import ModelBlock

class DIYModel(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, hidden_size: int, 
                 num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ModelBlock(hidden_size, num_heads, max_seq_len, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(self, input_ids, seq_lengths=None, past_key_values=None, use_cache=False):
        
        batch_size, seq_len = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        elif len(past_key_values) != self.num_layers:
            raise ValueError(f"past_key_values length ({len(past_key_values)}) must match num_layers ({self.num_layers})")

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 只在需要时构建 presents 列表
        presents = [] if (not self.training and use_cache) else None

        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            if self.training:
                hidden_states = layer(
                    hidden_states, seq_lengths=seq_lengths, cached=None, cached_pos=None
                )
            else:
                hidden_states, cached, cached_pos = layer(
                    hidden_states, 
                    seq_lengths=seq_lengths, 
                    cached=past_kv[0] if past_kv else None,
                    cached_pos=past_kv[1] if past_kv else None, 
                )
                if use_cache:
                    presents.append((cached, cached_pos))

        hidden_states = self.norm(hidden_states)

        if self.training:
            return hidden_states
        else:
            return (hidden_states, presents) if use_cache else hidden_states


class DIYForCausalLM(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, hidden_size: int, 
                 num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.model = DIYModel(vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, seq_lengths=None, past_key_values=None, use_cache=False):
        out = self.model(input_ids, seq_lengths=seq_lengths, past_key_values=past_key_values, 
                         use_cache=use_cache)

        if isinstance(out, tuple):
            hidden_states, presents = out
        else:
            hidden_states, presents = out, None

        logits = self.lm_head(hidden_states)

        if presents is not None:
            return logits, presents
        else:
            return logits


def trainer():
    vocab_size = 6400
    hidden_size = 256
    num_layers = 2
    num_heads = 4
    max_seq_len = 128
    batch_size, seq_len = 2, 10

    torch.manual_seed(42)
    model = DIYForCausalLM(vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 一次前向（forward 内部会 backward，这里先 zero_grad）
    model.train()

    for step in range(5):
        optimizer.zero_grad()
        logits = model(input_ids)

        logits_slice = logits[:, :-1, :].reshape(-1, vocab_size)
        labels_slice = input_ids[:, 1:].reshape(-1)

        loss = F.cross_entropy(logits_slice, labels_slice)
        loss.backward()
        optimizer.step()

        print(f"step {step + 1}, loss = {loss.item():.4f}")


def infer():
    vocab_size = 6400
    hidden_size = 256
    num_layers = 2
    num_heads = 4
    max_seq_len = 128

    torch.manual_seed(42)
    model = DIYForCausalLM(vocab_size, num_layers, hidden_size, num_heads, max_seq_len, dropout=0.1)
    model.eval()
    prompt = torch.tensor([[1, 42, 7, 0, 15]], dtype=torch.long)  # (1, 5)
    num_generate = 10

    print("=== 无 cache 生成（每次整段 forward）===")
    generated = prompt.clone()
    for step in range(num_generate):
        logits = model(generated)    # 每次都是整段
        next_logits = logits[:, -1, :]  # (1, V)
        next_token = next_logits.argmax(dim=-1)  # (1,)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        print(f"step {step}, generated token: {next_token.item()}")

    print("\n=== 有 cache 生成（增量解码）===")
    generated = prompt.clone()
    # 第一次：整段 prompt
    logits, presents = model(prompt, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    print(f"step 0, generated token: {next_token.item()}")
    
    # 后续每步：只送 1 个 token
    for step in range(num_generate - 1):
        new_token = next_token.unsqueeze(1) # (1, 1)
        logits, presents = model(new_token, past_key_values=presents, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        print(f"step {step + 1}, generated token: {next_token.item()}")


if __name__ == "__main__":
    infer()