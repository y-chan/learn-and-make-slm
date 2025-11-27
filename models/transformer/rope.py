from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
import torch


@jaxtyped(typechecker=beartype)
def rotate_half(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
    x1, x2 = x[..., 0], x[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)  # [..., dim/2, 2]
    return rotated.reshape(*x.shape[:-2], -1)  # [..., dim]


class RotaryPositionalEncoding(nn.Module):
    cos: Tensor  # (max_seq_len, dim)
    sin: Tensor  # (max_seq_len, dim)

    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryPositionalEncoding expects an even dimension.")

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Buffers move with the module and are optionally re-computed on demand.
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

        self._update_cache(seq_len=max_seq_len, device=torch.device("cpu"), dtype=torch.get_default_dtype())

    @jaxtyped(typechecker=beartype)
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

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "... seq_len {self.dim}"]) -> Float[Tensor, "... seq_len {self.dim}"]:  # noqa: F821
        seq_len = x.size(-2)
        self._update_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        cos = self.cos[:seq_len, :]  # (seq_len, dim)
        sin = self.sin[:seq_len, :]  # (seq_len, dim)

        return (x * cos) + (rotate_half(x) * sin)


if __name__ == "__main__":
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
    y = rope(x)
    assert y.shape == x.shape, f"y shape {y.shape} must be same as x shape {x.shape}"
