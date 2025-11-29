from torch import nn, Tensor
import torch
from models.basic.softmax import Softmax
from models.basic.linear import Linear
from typing import Optional
from jaxtyping import Float, Bool, Int, jaxtyped
from beartype import beartype as typechecker

from utils.mask import make_pad_mask

try:
    import xformers.ops as xops  # type: ignore
    import xformers.ops.fmha as fmha

    _HAS_XFORMERS = True
except Exception:
    xops = None  # type: ignore
    fmha = None  # type: ignore
    _HAS_XFORMERS = False


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.softmax = Softmax()
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    @jaxtyped(typechecker=typechecker)
    def xformers_forward(
        self,
        Q: Float[Tensor, "B H S D={self.d_k}"],
        K: Float[Tensor, "B H S D"],
        V: Float[Tensor, "B H S D"],
        seq_lens: Optional[Int[Tensor, "B"]] = None,
    ) -> Float[Tensor, "B H S D"]:
        """
        xformersの`memory_efficient_attention`を用いてSDPAを計算する。
        おそらく内部でFlash Attention 2を実行していると思われる。
        `memory_efficient_attention`を活用するため、バッチ次元を1次元にし、
        シーケンス次元で結合する"パッキング"という処理を行っている。
        パッキングに合わせて、パッキングされた複数のシーケンスの独立性を保証するために
        `BlockDiagonalMask`を導入している。
        最終的に、入力と同一形状のものが返ってくるようなシェイプ変換も行っている。
        """
        if not _HAS_XFORMERS:
            raise RuntimeError("xFormers is not available")

        B, *_ = Q.shape

        Q: Float[Tensor, "B S H D={self.d_k}"] = Q.transpose(1, 2)
        K: Float[Tensor, "B S H D={self.d_k}"] = K.transpose(1, 2)
        V: Float[Tensor, "B S H D={self.d_k}"] = V.transpose(1, 2)

        # Bは1になる、Sは要素ごとに異なる
        Q_list: list[Float[Tensor, "1 S_i H D={self.d_k}"]]
        K_list: list[Float[Tensor, "1 S_i H D={self.d_k}"]]
        V_list: list[Float[Tensor, "1 S_i H D={self.d_k}"]]

        if seq_lens is not None:
            # unsqueezeが不要なようにインデックスアクセスではなくスライスでアクセスする
            Q_list = [Q[i : i + 1, : seq_lens[i]] for i in range(B)]
            K_list = [K[i : i + 1, : seq_lens[i]] for i in range(B)]
            V_list = [V[i : i + 1, : seq_lens[i]] for i in range(B)]
        else:
            Q_list = [Q[i : i + 1] for i in range(B)]
            K_list = [K[i : i + 1] for i in range(B)]
            V_list = [V[i : i + 1] for i in range(B)]

        # S_total = sum(seq_lens)になる、バッチ分割していたものを一つの系列に結合している
        Q_reshaped: Float[Tensor, "1 S_total H D={self.d_k}"]
        K_reshaped: Float[Tensor, "1 S_total H D={self.d_k}"]
        V_reshaped: Float[Tensor, "1 S_total H D={self.d_k}"]
        # ref: https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.BlockDiagonalMask
        # BlockDiagonalMaskはseq_lensが[2, 3, 2]であればこのようなマスクを作成する
        # この方式であればメモリ効率の良いAttentionを計算できる
        # [[   0,    0, -inf, -inf, -inf, -inf, -inf],
        #  [   0,    0, -inf, -inf, -inf, -inf, -inf],
        #  [-inf, -inf,    0,     0,   0, -inf, -inf],
        #  [-inf, -inf,    0,     0,   0, -inf, -inf],
        #  [-inf, -inf,    0,     0,   0, -inf, -inf],
        #  [-inf, -inf, -inf, -inf, -inf,    0,    0],
        #  [-inf, -inf, -inf, -inf, -inf,    0,    0]]
        attn_bias, Q_reshaped, K_reshaped, V_reshaped = fmha.BlockDiagonalMask.from_tensor_lists_qkv(Q_list, K_list, V_list)

        out: Float[Tensor, "1 S_total H D={self.d_k}"] = xops.memory_efficient_attention(
            Q_reshaped, K_reshaped, V_reshaped, attn_bias=attn_bias, scale=float(self.scale)
        )  # type: ignore

        # 系列を結合していたものをバッチごとに分割する
        list_out: list[Float[Tensor, "1 S_i H D={self.d_k}"]] = attn_bias.split(out)
        # もとの形状に戻す
        padded_out: Float[Tensor, "B S H D={self.d_k}"] = torch.empty_like(Q)
        for i, out_elem in enumerate(list_out):
            if seq_lens is not None:
                padded_out[i][: seq_lens[i]] = out_elem
            else:
                padded_out[i] = out_elem
        padded_out: Float[Tensor, "B H S D={self.d_k}"] = padded_out.transpose(1, 2)
        return padded_out

    @jaxtyped(typechecker=typechecker)
    def reference_forward(
        self,
        Q: Float[Tensor, "B H S D={self.d_k}"],
        K: Float[Tensor, "B H S D"],
        V: Float[Tensor, "B H S D"],
        seq_lens: Optional[Int[Tensor, "B"]] = None,
    ) -> Float[Tensor, "B H S D"]:
        scores: Float[Tensor, "B H S S"] = (Q @ K.transpose(-2, -1)) * self.scale
        if seq_lens is not None:
            # make_pad_maskにmaxlenを明示的に渡す
            # seq_lensは学習に使うシーケンス長を表すので、必ずしも最大長が含まれるとは限らない
            maxlen = Q.size(-2)
            mask: Bool[Tensor, "B 1 1 S"] = (
                make_pad_mask(seq_lens, maxlen=maxlen).to(scores.device).unsqueeze(1).unsqueeze(1)
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = self.softmax(scores)
        output = attn_weights @ V
        return output

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        Q: Float[Tensor, "B H S D={self.d_k}"],
        K: Float[Tensor, "B H S D"],
        V: Float[Tensor, "B H S D"],
        seq_lens: Optional[Int[Tensor, "B"]] = None,
    ) -> Tensor:
        try:
            return self.xformers_forward(Q, K, V, seq_lens)
        except Exception:
            # 使えない状況（未対応のmaskやCPU/未実装カーネル等）は自前実装にフォールバック
            return self.reference_forward(Q, K, V, seq_lens)


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
        Q = (
            self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )  # (batch_size, n_heads, seq_len, d_k)
        K = (
            self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )  # (batch_size, n_heads, seq_len, d_k)
        V = (
            self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )  # (batch_size, n_heads, seq_len, d_k)

        attention = self.attention(Q, K, V, mask)
        attention = attention.transpose(1, 2)  # (batch_size, seq_len, n_heads, d_k)
        attention = attention.contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.linear_out(attention)  # (batch_size, seq_len, d_model)
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

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Tensor, "B S D={self.d_model}"], seq_lens: Optional[Int[Tensor, "B"]] = None
    ) -> Float[Tensor, "B S D"]:
        batch_size, seq_len, _ = x.size()
        Q: Float[Tensor, "B H={self.n_heads} S D"] = (
            self.linear_q(x).reshape(batch_size, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        )

        K: Float[Tensor, "B H_G={self.n_heads}//{self.n_groups} S D"] = (
            self.linear_k(x)
            .reshape(batch_size, seq_len, self.n_heads // self.n_groups, self.d_model // self.n_heads)
            .transpose(1, 2)
        )
        K: Float[Tensor, "B H_G 1 S D"] = K.unsqueeze(2)
        K: Float[Tensor, "B H_G G S D"] = K.expand(-1, -1, self.n_groups, -1, -1)
        K: Float[Tensor, "B H S D"] = K.reshape(batch_size, self.n_heads, seq_len, self.d_model // self.n_heads)

        V: Float[Tensor, "B H_G S D"] = (
            self.linear_v(x)
            .reshape(batch_size, seq_len, self.n_heads // self.n_groups, self.d_model // self.n_heads)
            .transpose(1, 2)
        )
        V: Float[Tensor, "B H_G 1 S D"] = V.unsqueeze(2)
        V: Float[Tensor, "B H_G G S D"] = V.expand(-1, -1, self.n_groups, -1, -1)
        V: Float[Tensor, "B H S D"] = V.reshape(batch_size, self.n_heads, seq_len, self.d_model // self.n_heads)

        attention = self.attention(Q, K, V, seq_lens)
        attention = attention.transpose(1, 2)  # (batch_size, seq_len, n_heads, d_k)
        attention = attention.contiguous().reshape(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.linear_out(attention)  # (batch_size, seq_len, d_model)
        return output
