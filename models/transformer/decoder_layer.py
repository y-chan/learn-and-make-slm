from torch import nn, Tensor
from models.transformer.attention import GroupedQueryAttention
from models.transformer.feed_forward import FeedForwardSwiGLU
from models.basic.layer_norm import LayerNorm
from jaxtyping import Float, Int


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, n_groups: int, rope_scale_factor: float = 1.0, use_sigmoid_gate: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        # masked multi-head self-attention(grouped query attention)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_groups=n_groups,
            use_rope=True,
            rope_scale_factor=rope_scale_factor,
            use_sigmoid_gate=use_sigmoid_gate,
        )
        self.norm1 = LayerNorm(d_model)

        # feed-forward
        self.ffn = FeedForwardSwiGLU(d_model=d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S D"]:
        # masked multi-head self-attention
        residual = x
        x = self.norm1(x)
        attn_output: Float[Tensor, "B S D"] = self.self_attn(x, seq_lens)
        x = residual + attn_output

        # feed-forward
        residual = x
        x = self.norm2(x)
        ffn_output: Float[Tensor, "B S D"] = self.ffn(x)
        x = residual + ffn_output

        return x
