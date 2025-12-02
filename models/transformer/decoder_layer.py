from typing import Optional

from torch import nn, Tensor
from jaxtyping import Float, Int

from models.transformer.attention import MultiHeadAttention
from models.transformer.feed_forward import FeedForwardSwish, FeedForwardSwiGLU
from models.basic.layer_norm import LayerNorm


class GPT2DecoderLayer(nn.Module):
    """
    概ねGPT-2のDecoder Layerを再現している
    異なるのはFeed Forward BlockがGELUではなくSwishを使っている点
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model

        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads
        )
        self.layer_norm1 = LayerNorm([d_model])
        self.feed_forward_block = FeedForwardSwish(d_model)
        self.layer_norm2 = LayerNorm([d_model])

    def forward(
        self,
        x: Float[Tensor, "B S D={self.d_model}"],
        seq_lens: Optional[Int[Tensor, "B"]] = None
    ) -> Float[Tensor, "B S D"]:
        residual = x
        x = self.layer_norm1(x)
        x = residual + self.multi_head_attention(x, seq_lens)

        residual = x
        x = self.layer_norm2(x)
        x = residual + self.feed_forward_block(x)

        return x


class GPTOSSDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
