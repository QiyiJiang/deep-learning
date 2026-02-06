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

    def forward(self, input_ids, seq_lengths=None, past_key_values=None, use_cache=False, is_training=True):
        batch_size, seq_len = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        elif len(past_key_values) != self.num_layers:
            raise ValueError(f"past_key_values length ({len(past_key_values)}) must match num_layers ({self.num_layers})")

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 只在需要时构建 presents 列表
        presents = [] if (not is_training and use_cache) else None

        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            if is_training:
                hidden_states = layer(
                    hidden_states, seq_lengths=seq_lengths, cached=None, cached_pos=None, is_training=True
                )
            else:
                hidden_states, cached, cached_pos = layer(
                    hidden_states, 
                    seq_lengths=seq_lengths, 
                    cached=past_kv[0] if past_kv else None,
                    cached_pos=past_kv[1] if past_kv else None, 
                    is_training=False
                )
                if use_cache:
                    presents.append((cached, cached_pos))

        hidden_states = self.norm(hidden_states)

        if is_training:
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

    def forward(self, input_ids):

        out = self.model(input_ids)

        if isinstance(out, tuple):
            hidden_states, presents = out
        else:
            hidden_states, presents = out, None

        logits = self.lm_head(hidden_states)

        if presents is not None:
            return logits, presents
        else:
            return logits


if __name__ == "__main__":
    # 简单测试：前向 + 看 logits / loss，再跑几步看 loss 是否下降
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
    optimizer.zero_grad()
    loss = model(input_ids)
    print(f"logits.shape = {input_ids.shape}, loss = {loss.item():.4f}")

    # 对同一批数据跑几步，看 loss 是否下降
    for step in range(5):
        optimizer.zero_grad()
        loss = model(input_ids)
        optimizer.step()
        print(f"step {step + 1}, loss = {loss.item():.4f}")