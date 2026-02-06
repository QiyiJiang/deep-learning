from cProfile import label
from math import log
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import checkpoint
from typing import Optional, Tuple, List, Dict, Union
from model.modules.RMSNorm import RMSNorm
from model.modules.ModelBlock import ModelBlock
from model.modules.modelconfig import DIYCofig

class DIYModel(nn.Module):
    """DIY 模型主干：embed → ModelBlock × N → RMSNorm，输出 hidden_states。"""
    def __init__(self, config: DIYCofig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            ModelBlock(config) 
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config)

    def forward(
        self, 
        input_ids: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[Dict[str, torch.Tensor], int]]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[Dict[str, torch.Tensor], int]]]]:
        """
        前向传播。
        
        Args:
            input_ids: Token IDs，shape (batch_size, seq_len)
            seq_lengths: 每个样本的有效长度，shape (batch_size,)，None 表示无 padding
            past_key_values: KV cache，每层一个 (cached_dict, cached_pos) tuple 或 None
            use_cache: 是否返回 presents（用于增量解码）
        
        Returns:
            训练时返回 hidden_states，shape (batch_size, seq_len, hidden_size)
            推理且 use_cache=True 时返回 (hidden_states, presents)
        """
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
    """DIY 因果语言模型：DIYModel + lm_head，用于训练和生成。"""
    
    def __init__(self, config: DIYCofig):
        super().__init__()
        
        self.vocab_size = config.vocab_size
        self.model = DIYModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[Dict[str, torch.Tensor], int]]]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[Dict[str, torch.Tensor], int]]]]:
        """
        前向传播，返回 logits。
        
        Args:
            input_ids: Token IDs，shape (batch_size, seq_len)
            attention_mask: Attention mask，1=有效，0=padding，shape (batch_size, seq_len)
            seq_lengths: 每个样本的有效长度，shape (batch_size,)，与 attention_mask 二选一
            past_key_values: KV cache，用于增量解码
            use_cache: 是否返回 presents（用于增量解码）
        
        Returns:
            训练时返回 logits，shape (batch_size, seq_len, vocab_size)
            推理且 use_cache=True 时返回 (logits, presents)
        """
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1)

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
    config = DIYCofig()
    batch_size, seq_len = 2, 10

    torch.manual_seed(42)
    model = DIYForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 一次前向（forward 内部会 backward，这里先 zero_grad）
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
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, "diy_checkpoint.pth")

    checkpoint = torch.load("diy_checkpoint.pth")
    config = DIYCofig(**checkpoint["config"])
    model2 = DIYForCausalLM(config)
    model2.load_state_dict(checkpoint["state_dict"])

    for step in range(5, 10):
        optimizer.zero_grad()
        logits = model(input_ids)

        logits_slice = logits[:, :-1, :].reshape(-1, config.vocab_size)
        labels_slice = input_ids[:, 1:].reshape(-1)

        loss = F.cross_entropy(logits_slice, labels_slice)
        loss.backward()
        optimizer.step()

        print(f"step {step + 1}, loss = {loss.item():.4f}")

def infer():
    config = DIYCofig()

    torch.manual_seed(42)
    model = DIYForCausalLM(config)
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
    trainer()