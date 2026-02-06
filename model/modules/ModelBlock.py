import torch
from torch import nn
from model.modules.Attention import FlashAttentionFusedAttention
from model.modules.RMSNorm import RMSNorm
from typing import Optional, Tuple, List, Dict, Union
from model.modules.FeedForward import GatedFeedForward
from model.modules.modelconfig import DIYCofig


class ModelBlock(nn.Module):
    def __init__(self, config: DIYCofig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout

        self.attention = FlashAttentionFusedAttention(config)
        self.attention_norm = RMSNorm(config)
        self.feedforward_norm = RMSNorm(config)
        self.feedforward = GatedFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        cached: Optional[Dict[str, torch.Tensor]] = None,
        cached_pos: Optional[int] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """
        前向传播。
        
        Args:
            x: Input tensor，shape (batch_size, seq_len, hidden_size)
            seq_lengths: 每个样本的有效长度，shape (batch_size,)，用于 padding mask
            cached: KV cache dict，包含 "k" 和 "v"，用于增量解码
            cached_pos: Cache 位置，表示当前缓存到第几个位置
        
        Returns:
            训练时返回 x，shape (batch_size, seq_len, hidden_size)
            推理时返回 (x, cached, cached_pos) tuple
        """
        # Pre-norm attention
        residual = x
        x = self.attention_norm(x)
        att_output, cached, cached_pos = self.attention(
            x, 
            seq_lengths=seq_lengths, 
            cached=cached, 
            cached_pos=cached_pos, 
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin
        )
        x = residual + att_output
        
        # Pre-norm feedforward
        residual = x
        x = self.feedforward_norm(x)
        x = self.feedforward(x)
        x = residual + x
        
        # 训练模式下只返回输出，推理模式下返回输出和缓存
        if self.training:
            return x
        else:
            return x, cached, cached_pos
