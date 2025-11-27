from torch import nn, Tensor
import torch
from models.basic.softmax import Softmax
from models.basic.linear import Linear
from typing import Optional

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.softmax = Softmax()
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = self.softmax(scores)
        output = attn_weights @ V
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model // n_heads)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        Q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        K = self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        V = self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)

        attention = self.attention(Q, K, V, mask)
        attention = attention.transpose(1, 2) # (batch_size, seq_len, n_heads, d_k)
        attention = attention.contiguous().view(batch_size, seq_len, self.d_model) # (batch_size, seq_len, d_model)
        output = self.linear_out(attention) # (batch_size, seq_len, d_model)
        return output

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_groups: int):
        super().__init__()
        assert n_heads % n_groups == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_per_group = n_heads // n_groups
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model // n_groups)
        self.linear_v = Linear(d_model, d_model // n_groups)
        self.linear_out = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model // n_heads)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        Q = self.linear_q(x).reshape(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)

        K = self.linear_k(x).reshape(batch_size, seq_len, self.n_heads//self.n_groups, self.d_model // self.n_heads).transpose(1, 2) # (B, H_G, S, D)
        K = K.unsqueeze(2) # (B, H_G, 1, S, D)
        K = K.expand(-1, -1, self.n_groups, -1, -1) # (B, H_G, G, S, D)
        K = K.reshape(batch_size, self.n_heads, seq_len, self.d_model // self.n_heads) # (B, H, S, D)

        V = self.linear_v(x).reshape(batch_size, seq_len, self.n_heads//self.n_groups, self.d_model // self.n_heads).transpose(1, 2) # (B, H_G, S, D)
        V = V.unsqueeze(2) # (B, H_G, 1, S, D)
        V = V.expand(-1, -1, self.n_groups, -1, -1) # (B, H_G, G, S, D)
        V = V.reshape(batch_size, self.n_heads, seq_len, self.d_model // self.n_heads) # (B, H, S, D)

        attention = self.attention(Q, K, V, mask)
        attention = attention.transpose(1, 2) # (batch_size, seq_len, n_heads, d_k)
        attention = attention.contiguous().reshape(batch_size, seq_len, self.d_model) # (batch_size, seq_len, d_model)
        output = self.linear_out(attention) # (batch_size, seq_len, d_model)
        return output