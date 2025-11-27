from beartype import beartype
from jaxtyping import jaxtyped, Float
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
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Build rotate matrix with max size at initialization
        _cos, _sin = self._compute_rotates(max_seq_len=max_seq_len, dim=dim, device=torch.device("cpu"))
        self.cos = _cos
        self.sin = _sin

    @jaxtyped(typechecker=beartype)
    def _compute_rotates(
        self, *, max_seq_len: int, dim: int, device: torch.device
    ) -> tuple[Float[Tensor, "{max_seq_len} {dim}"], Float[Tensor, "{max_seq_len} {dim}"]]:  # noqa: F821
        base = 10000
        vec = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))  # (dim/2,)
        thetas = (
            vec.unsqueeze(1)  # (dim/2, 1)
            .expand(-1, 2)  # (dim/2, 2)
            .reshape(-1)  # (dim,)
        )

        pos = torch.arange(0, max_seq_len, dtype=torch.float)  # (max_seq_len,)
        rotate = torch.outer(pos, thetas)  # (max_seq_len, dim)

        return torch.cos(rotate), torch.sin(rotate)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "... seq_len {self.dim}"]) -> Float[Tensor, "... seq_len {self.dim}"]:  # noqa: F821
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            # Expand cache automatically
            _cos, _sin = self._compute_rotates(max_seq_len=seq_len, dim=self.dim, device=x.device)
            self.cos = _cos
            self.sin = _sin
            self.max_seq_len = seq_len

        cos = self.cos[:seq_len, :]  # (seq_len, dim)
        sin = self.sin[:seq_len, :]  # (seq_len, dim)

        return (x * cos) + (rotate_half(x) * sin)


if __name__ == "__main__":
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
    y = rope(x)
    assert y.shape == x.shape, f"y shape {y.shape} must be same as x shape {x.shape}"
