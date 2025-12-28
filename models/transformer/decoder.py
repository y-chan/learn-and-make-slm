from typing import Final
import sys
import torch
import tiktoken

from torch import nn, Tensor
from models.basic.layer_norm import LayerNorm
from models.transformer.attention import GroupedQueryAttention, MultiHeadAttention, _INTERNAL_INITIAL_CACHE_INDEX
from models.transformer.decoder_layer import GPT2DecoderLayer, GPTOSSDecoderLayer
from models.basic.linear import Linear
from models.basic.softmax import Softmax
from jaxtyping import Float, Int, Bool
from models.basic.embedding import Embedding
from models.transformer.positional_encoding import PositionalEncoding
from utils.mask import make_non_pad_mask


CACHEABLE_MODULES: Final = (MultiHeadAttention, GroupedQueryAttention)


class DecoderBase(nn.Module):
    def __init__(self, n_vocab: int, d_model: int, end_token_id: int, *, enable_cache: bool):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.end_token_id = end_token_id
        self.enable_cache = enable_cache

        self.softmax = Softmax()

    def _activate_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, CACHEABLE_MODULES):
                module._internal_activate_cache()

    def _invalidate_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, CACHEABLE_MODULES):
                module._internal_invalidate_cache()

    def forward(
        self,
        x: Int[Tensor, "B S"],
        seq_lens: Int[Tensor, "B"] | None = None,  # noqa: F821
    ) -> Float[Tensor, "B S V={self.n_vocab}"]:
        raise NotImplementedError()

    @torch.no_grad()
    def infer(
        self,
        starts: Int[Tensor, "1 S_in"],
        max_token_count: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> Int[Tensor, "1 S_out"]:
        try:
            if self.enable_cache:
                self._activate_caches()

            assert starts.size(0) == 1, "starts must be a 1D tensor"
            x: Tensor = starts
            count: int = 0

            if self.enable_cache and x.size(1) > 1:
                # キャッシュを使う場合は、一番最後のトークンを除いてキャッシュを準備する
                self(x.detach().clone()[:, :-1])

            if tokenizer is not None:
                decoded = tokenizer.decode(starts[0].tolist())
                print("".join(decoded), end="", flush=True)

            def loop_condition(count: int) -> bool:
                if max_token_count is not None:
                    return count < max_token_count
                return True

            while loop_condition(count=count):
                if self.enable_cache:
                    # キャッシュを使う場合は、一番最後のトークンのみ渡す
                    output = self(x.detach().clone()[:, -1:])
                else:
                    output = self(x.detach().clone())
                if temperature == 0.0:
                    # argmax: Greedy Decodingによる最も確率の高いトークンを選択
                    next_token = output.argmax(dim=-1)[:, -1:]
                else:
                    output_prob = self.softmax(output / temperature)
                    # 最後の位置の確率分布からサンプリング
                    next_token_probs = output_prob[:, -1, :]  # [B, V]
                    next_token_indices: Tensor | None = None
                    if top_k is not None:
                        next_token_probs, next_token_indices = torch.topk(next_token_probs, top_k, dim=-1)
                        next_token_probs = next_token_probs / next_token_probs.sum(dim=-1, keepdim=True)  # 再正規化
                    # torch.multinomialでカテゴリカル分布からサンプリング
                    next_token = torch.multinomial(next_token_probs, num_samples=1)  # [B, 1]
                    if next_token_indices is not None:
                        next_token = torch.gather(next_token_indices, dim=-1, index=next_token)

                x = torch.cat([x, next_token], dim=-1)
                count += 1
                if next_token[0, 0] == self.end_token_id:
                    break
                if tokenizer is not None:
                    next_token = tokenizer.decode_tokens_bytes([next_token.item()])
                    for byte in next_token:
                        sys.stdout.buffer.write(byte)
                    sys.stdout.flush()
            return x
        finally:
            if self.enable_cache:
                self._invalidate_caches()

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

    def __init__(
        self, n_vocab: int, n_layers: int, d_model: int, n_heads: int, end_token_id: int, *, enable_cache: bool = False
    ):
        super().__init__(n_vocab, d_model, end_token_id, enable_cache=enable_cache)

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
        positional_offset = 0
        # FIXME: より良い実装を模索する
        if self.enable_cache:
            _active_cache = self.decoder_layers[0].multi_head_attention._active_cache
            if _active_cache is not None and _active_cache != _INTERNAL_INITIAL_CACHE_INDEX:
                assert x.size(1) == 1, "When using cache, x must have sequence length 1"

            positional_offset = self.decoder_layers[0].multi_head_attention._current_seq_len

        x: Float[Tensor, "B S D={self.d_model}"] = self.embedding(x)
        x = self.positional_encoding(x, positional_offset)

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

    def __init__(
        self,
        n_vocab: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_groups: int,
        end_token_id: int,
        rope_scale_factor: float = 1.0,
        *,
        enable_cache: bool = False,
    ):
        super().__init__(n_vocab, d_model, end_token_id, enable_cache=enable_cache)

        self.embedding = Embedding(n_vocab, d_model)
        self.layers = nn.ModuleList(
            [
                GPTOSSDecoderLayer(d_model=d_model, n_heads=n_heads, n_groups=n_groups, rope_scale_factor=rope_scale_factor)
                for _ in range(n_layers)
            ]
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
