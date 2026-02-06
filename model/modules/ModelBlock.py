import torch
from torch import nn
from model.modules.Attention import FlashAttentionFusedAttention
from model.modules.RMSNorm import RMSNorm
from model.modules.FeedForward import GatedFeedForward


class ModelBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.attention = FlashAttentionFusedAttention(self.hidden_size, self.num_heads, self.max_seq_len, self.dropout)
        self.attention_norm = RMSNorm(self.hidden_size)
        self.feedforward_norm = RMSNorm(self.hidden_size)
        self.feedforward = GatedFeedForward(self.hidden_size, 4*self.hidden_size, self.dropout)

    def forward(self, x, seq_lengths=None, cached=None, cached_pos=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            seq_lengths: Optional sequence lengths for padding mask
            cached: Optional KV cache dict for inference
            cached_pos: Optional cache position for inference
            is_training: Whether in training mode
        
        Returns:
            If is_training=True: x (output tensor)
            If is_training=False: (x, cached, cached_pos) tuple for incremental decoding
        """
        # Pre-norm attention
        residual = x
        x = self.attention_norm(x)
        att_output, cached, cached_pos = self.attention(
            x, 
            seq_lengths=seq_lengths, 
            cached=cached, 
            cached_pos=cached_pos, 
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
