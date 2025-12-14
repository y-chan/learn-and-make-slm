from torch import nn, Tensor
from models.transformer.attention import GroupedQueryAttention
from models.transformer.feed_forward import FeedForwardSwiGLU
from models.basic.layer_norm import LayerNorm
from jaxtyping import Float, Int
from typing import Optional


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_groups: int):
        super().__init__()
        self.d_model = d_model
        # masked multi-head self-attention(grouped query attention)
        self.self_attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_groups=n_groups)
        self.norm1 = LayerNorm(d_model)

        # feed-forward
        self.ffn = FeedForwardSwiGLU(d_model=d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        seq_lens: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
    ) -> Float[Tensor, "B S D"]:
        # masked multi-head self-attention
        attn_output: Float[Tensor, "B S D"] = self.self_attn(x, seq_lens)
        x = self.norm1(x + attn_output)

        # feed-forward
        ffn_output: Float[Tensor, "B S D"] = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
