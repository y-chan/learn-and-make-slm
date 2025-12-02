from typing import Optional

from torch import nn, Tensor
from jaxtyping import Float, Bool, Int

from models.basic.embedding import Embedding
from models.basic.layer_norm import LayerNorm
from models.basic.linear import Linear
from models.basic.softmax import Softmax
from models.transformer.positional_encoding import PositionalEncoding
from models.transformer.decoder_layer import GPT2DecoderLayer
from utils.mask import make_non_pad_mask


class GPT2Decoder(nn.Module):
    """
    概ねGPT-2なTransformerのDecoder
    Decoder Layerの実装に一部差異がある
    """
    def __init__(self, n_vocab: int, n_layers: int, d_model: int, n_heads: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model

        self.embedding = Embedding(n_vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([
            GPT2DecoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.layer_norm = LayerNorm([d_model])
        self.linear_out = Linear(d_model, n_vocab)
        self.softmax = Softmax()

    def forward(
        self,
        x: Int[Tensor, "B S"],
        seq_lens: Optional[Int[Tensor, "B"]] = None
    ) -> Float[Tensor, "B S V={self.n_vocab}"]:
        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, seq_lens)

        x = self.layer_norm(x)
        y: Float[Tensor, f"B S V"] = self.linear_out(x)
        y = self.softmax(y)
        return y

    def loss(
        self,
        pred_y: Float[Tensor, "B S V={self.n_vocab}"],
        target_y_index: Int[Tensor, "B S"],
        seq_lens: Optional[Int[Tensor, "B"]] = None
    ) -> Float[Tensor, "1"]:
        if seq_lens:
            non_pad_mask: Bool[Tensor, "B S"] = make_non_pad_mask(seq_lens, pred_y.size(-1))

            # 関係ないところに逆伝播が流れないようにマスクする
            pred_y: Float[Tensor, "B V S"] = pred_y.transpose(-1, -2) * non_pad_mask

        loss: Float[Tensor, "1"] = nn.functional.cross_entropy(pred_y, target_y_index)
        return loss


class GPTOSSDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_layers: int, d_model: int, n_heads: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
