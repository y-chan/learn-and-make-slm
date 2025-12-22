import torch
import tiktoken

from torch import nn, Tensor
from models.basic.layer_norm import LayerNorm
from models.transformer.decoder_layer import GPT2DecoderLayer, GPTOSSDecoderLayer
from models.basic.linear import Linear
from models.basic.softmax import Softmax
from jaxtyping import Float, Int, Bool
from models.basic.embedding import Embedding
from models.transformer.positional_encoding import PositionalEncoding
from utils.mask import make_non_pad_mask


class DecoderBase(nn.Module):
    def __init__(self, n_vocab: int, d_model: int, end_token_id: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.end_token_id = end_token_id

        self.softmax = Softmax()

    def forward(
        self,
        x: Int[Tensor, "B S"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S V={self.n_vocab}"]:
        raise NotImplementedError()

    @torch.no_grad()
    def infer(
        self,
        starts: Int[Tensor, "1 S"],
        max_token_count: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> Int[Tensor, "1 S"]:
        assert starts.size(0) == 1, "starts must be a 1D tensor"
        x = starts
        count = 0
        if tokenizer is not None:
            decoded = tokenizer.decode(starts[0].tolist())
            print("".join(decoded), end="", flush=True)

        def loop_condition(count: int) -> bool:
            if max_token_count is not None:
                return count < max_token_count
            return True

        while loop_condition(count=count):
            output = self(x.detach().clone())
            # TODO: temperatureなどを考慮したサンプリングを実装する
            # TODO: KVキャッシュを考慮した形にする
            if temperature == 0.0:
                # argmax: Greedy Decodingによる最も確率の高いトークンを選択
                next_token = output.argmax(dim=-1)[:, -1:]
            else:
                output_prob = self.softmax(output / temperature)
                # 最後の位置の確率分布からサンプリング
                next_token_probs = output_prob[:, -1, :]  # [B, V]
                next_token_indices = None
                if top_k is not None:
                    next_token_probs, next_token_indices = torch.topk(next_token_probs, top_k, dim=-1)
                    next_token_probs = next_token_probs / next_token_probs.sum(dim=-1, keepdim=True)  # 再正規化
                # torch.multinomialでカテゴリカル分布からサンプリング
                next_token = torch.multinomial(next_token_probs, num_samples=1)  # [B, 1]
                if output_indices is not None:
                    next_token = torch.gather(next_token_indices, dim=-1, index=next_token)

            x = torch.cat([x, next_token], dim=-1)
            count += 1
            if next_token[0, 0] == self.end_token_id:
                break
            if tokenizer is not None:
                next_token = tokenizer.decode([next_token.item()])
                print(next_token[0], end="", flush=True)
        return x

    def loss(
        self,
        pred_y: Float[Tensor, "B S V={self.n_vocab}"],
        target_y_index: Int[Tensor, "B S"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "1"]:
        pred_y: Float[Tensor, "B V S"] = pred_y.transpose(-1, -2)

        # cross_entropyはreduction='none'で各位置のlossを計算し、
        # 後でmaskを適用して平均を取る
        loss: Float[Tensor, "B S"] = nn.functional.cross_entropy(pred_y, target_y_index, reduction="none")

        if seq_lens is not None:
            non_pad_mask: Bool[Tensor, "B S"] = make_non_pad_mask(seq_lens, maxlen=loss.size(-1)).to(loss.device)
            # パディング位置のlossを0にする
            loss = loss * non_pad_mask
            # 有効な位置の平均を取る（division by zeroを防ぐため最小値1を保証）
            loss = loss.sum() / torch.clamp(non_pad_mask.sum(), min=1.0)
        else:
            loss = loss.mean()

        return loss


class GPT2Decoder(DecoderBase):
    """
    概ねGPT-2なTransformerのDecoder
    Decoder Layerの実装に一部差異がある
    """

    def __init__(self, n_vocab: int, n_layers: int, d_model: int, n_heads: int, end_token_id: int):
        super().__init__(n_vocab, d_model, end_token_id)

        self.embedding = Embedding(n_vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([GPT2DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.layer_norm = LayerNorm([d_model])
        self.linear_out = Linear(d_model, n_vocab)

    def forward(
        self,
        x: Int[Tensor, "B S"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S V={self.n_vocab}"]:
        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, seq_lens)

        x = self.layer_norm(x)
        y: Float[Tensor, "B S V"] = self.linear_out(x)
        return y


class GPTOSSDecoder(DecoderBase):
    """
    概ねGPT-OSSなTransformerのDecoder
    Decoder Layerの実装に一部差異がある
    """

    def __init__(self, n_vocab: int, n_layers: int, d_model: int, n_heads: int, n_groups: int, end_token_id: int):
        super().__init__(n_vocab, d_model, end_token_id)

        self.embedding = Embedding(n_vocab, d_model)
        self.layers = nn.ModuleList(
            [GPTOSSDecoderLayer(d_model=d_model, n_heads=n_heads, n_groups=n_groups) for _ in range(n_layers)]
        )
        self.linear_out = Linear(d_model, n_vocab)

    def forward(
        self,
        x: Int[Tensor, "B S"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S V={self.n_vocab}"]:
        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)

        for layer in self.layers:
            x = layer(x, seq_lens)

        output = self.linear_out(x)
        return output
