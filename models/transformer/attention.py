from torch import nn, Tensor
import torch
from models.basic.softmax import Softmax
from models.basic.linear import Linear

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.softmax = Softmax()

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: bool = False) -> Tensor:
        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores = scores.masked_fill(mask, float('-inf')) if mask else scores
        attn_weights = self.softmax(scores)
        output = attn_weights @ V
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model // n_heads)

    def forward(self, x: Tensor, mask: bool = False) -> Tensor:
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        attention = self.attention(Q, K, V, mask)
        output = self.linear_out(attention)
        return output