from jaxtyping import Float
from torch import Tensor, nn
import torch


def rotate_half(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
    x1, x2 = x[..., 0], x[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)  # [..., dim/2, 2]
    return rotated.reshape(*x.shape[:-2], -1)  # [..., dim]


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryPositionalEncoding expects an even dimension.")

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Buffers move with the module and are optionally re-computed on demand.
        self.cos: Tensor  # (max_seq_len, dim)
        self.sin: Tensor  # (max_seq_len, dim)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

        self._update_cache(seq_len=max_seq_len, device=torch.device("cpu"), dtype=torch.get_default_dtype())

    def _compute_rotates(
        self, *, max_seq_len: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Float[Tensor, "{max_seq_len} {dim}"], Float[Tensor, "{max_seq_len} {dim}"]]:  # noqa: F821
        base = 10_000
        vec = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))  # (dim/2,)
        thetas = (
            vec.unsqueeze(1)  # (dim/2, 1)
            .expand(-1, 2)  # (dim/2, 2)
            .reshape(-1)  # (dim,)
        )

        pos = torch.arange(0, max_seq_len, dtype=dtype, device=device)  # (max_seq_len,)
        rotate = torch.outer(pos, thetas)  # (max_seq_len, dim)

        return torch.cos(rotate), torch.sin(rotate)

    def _update_cache(self, *, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        needs_rebuild = (
            self.cos.numel() == 0  # initial build
            or seq_len > self.max_seq_len
            or self.cos.device != device
            or self.cos.dtype != dtype
        )
        if not needs_rebuild:
            return

        target_len = max(seq_len, self.max_seq_len)
        _cos, _sin = self._compute_rotates(max_seq_len=target_len, dim=self.dim, device=device, dtype=dtype)
        self.cos = _cos
        self.sin = _sin
        self.max_seq_len = target_len

    def apply_rotary_pos_emb(
        self,
        x: Float[Tensor, "... seq_len {self.dim}"],
        position_offset: int = 0,  # noqa: F821
    ) -> Float[Tensor, "... seq_len {self.dim}"]:  # noqa: F821
        """
        Apply rotary positional encoding with optional position offset.

        Parameters
        ----------
        x : Float[Tensor, "... seq_len {self.dim}"]
            Input tensor
        position_offset : int, optional
            Absolute position offset for cached inference, by default 0

        Returns
        -------
        Float[Tensor, "... seq_len {self.dim}"]
            Tensor with RoPE applied
        """
        seq_len = x.size(-2)
        total_len = position_offset + seq_len
        self._update_cache(seq_len=total_len, device=x.device, dtype=x.dtype)

        cos = self.cos[position_offset:total_len, :]  # (seq_len, dim)
        sin = self.sin[position_offset:total_len, :]  # (seq_len, dim)

        return (x * cos) + (rotate_half(x) * sin)

    def forward(self, x: Float[Tensor, "... seq_len {self.dim}"]) -> Float[Tensor, "... seq_len {self.dim}"]:  # noqa: F821
        return self.apply_rotary_pos_emb(x, position_offset=0)
