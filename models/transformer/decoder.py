from torch import nn, Tensor
from models.transformer.decoder_layer import DecoderLayer
from models.basic.linear import Linear
from jaxtyping import Float, Int, Bool
from typing import Optional
from models.basic.embedding import Embedding
from utils.mask import make_non_pad_mask


class Decoder(nn.Module):
    def __init__(self, n_vocab: int, n_layers: int, d_model: int, n_heads: int, n_groups: int, end_token_id: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.end_token_id = end_token_id

        self.embedding = Embedding(n_vocab, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, n_heads=n_heads, n_groups=n_groups) for _ in range(n_layers)]
        )
        self.linear_out = Linear(d_model, n_vocab)

    def forward(
        self, x: Int[Tensor, "B S"], seq_lens: Optional[Int[Tensor, "B"]] = None
    ) -> Float[Tensor, "B S V={self.n_vocab}"]:
        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)

        for layer in self.layers:
            x = layer(x, seq_lens)

        output = self.linear_out(x)
        return output

    def infer(self):
        raise NotImplementedError("Decoder.infer is not implemented yet.")

    def loss(
        self,
        pred_y: Float[Tensor, "B S V={self.n_vocab}"],
        target_y_index: Int[Tensor, "B S"],
        seq_lens: Optional[Int[Tensor, "B"]] = None,
    ) -> Float[Tensor, "1"]:
        pred_y: Float[Tensor, "B V S"] = pred_y.transpose(-1, -2)
        if seq_lens is not None:
            non_pad_mask: Bool[Tensor, "B S"] = make_non_pad_mask(seq_lens, maxlen=pred_y.size(-1))

            pred_y = pred_y * non_pad_mask

        loss: Float[Tensor, "1"] = nn.functional.cross_entropy(pred_y, target_y_index)
        return loss
