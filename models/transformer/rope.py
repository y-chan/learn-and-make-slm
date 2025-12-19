from jaxtyping import Float
from torch import Tensor, nn
import torch
import math


def rotate_half(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
    x1, x2 = x[..., 0], x[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)  # [..., dim/2, 2]
    return rotated.reshape(*x.shape[:-2], -1)  # [..., dim]


class YaRNRotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        original_max_seq_len: int = 2048,
        base: float = 10_000,
        alpha: float = 1.0,
        beta: float = 32.0,
        scale_factor: float = 1.0,
        enable_dynamic_scaling: bool = False,
    ) -> None:
        """
        YaRN (Yet another RoPE extensioN method) implementation.
        
        Args:
            dim: Dimension of the embedding (must be even)
            max_seq_len: Maximum sequence length for pre-computation
            original_max_seq_len: Original pre-training sequence length
            base: Base value for frequency computation (default: 10000)
            alpha: Lower bound for NTK-by-parts ramp function (default: 1.0)
            beta: Upper bound for NTK-by-parts ramp function (default: 32.0)
            scale_factor: Context extension scale factor s = L'/L (default: 1.0 for no extension)
            enable_dynamic_scaling: Whether to use dynamic scaling at inference time
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("YaRNRotaryPositionalEncoding expects an even dimension.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.scale_factor = scale_factor
        self.enable_dynamic_scaling = enable_dynamic_scaling

        # Compute attention temperature scaling
        # sqrt(1/t) = 0.1 * ln(s) + 1
        self.attention_scale = 0.1 * math.log(max(scale_factor, 1.0)) + 1.0

        # Buffers move with the module and are optionally re-computed on demand.
        self.cos: Tensor  # (max_seq_len, dim)
        self.sin: Tensor  # (max_seq_len, dim)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

        self._update_cache(
            seq_len=max_seq_len,
            device=torch.device("cpu"),
            dtype=torch.get_default_dtype(),
            scale_factor=scale_factor,
        )

    def _compute_ramp_function(self, r: Float[Tensor, "dim/2"]) -> Float[Tensor, "dim/2"]:
        """
        Compute the ramp function γ(r) for NTK-by-parts interpolation.
        
        γ(r) = 0 if r < α
             = 1 if r > β
             = (r - α)/(β - α) otherwise
        """
        gamma = torch.clamp((r - self.alpha) / (self.beta - self.alpha), min=0.0, max=1.0)
        return gamma

    def _compute_rotates(
        self,
        *,
        max_seq_len: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        scale_factor: float,
    ) -> tuple[Float[Tensor, "{max_seq_len} {dim}"], Float[Tensor, "{max_seq_len} {dim}"]]:  # noqa: F821
        """
        Compute cos and sin matrices with YaRN interpolation.
        """
        # Compute dimension indices (only even dimensions: 0, 2, 4, ...)
        d_indices = torch.arange(0, dim, 2, dtype=dtype, device=device)  # (dim/2,)
        
        # Compute wavelengths: λ_d = 2π * base^(2d/dim)
        wavelengths = 2 * math.pi * (self.base ** (d_indices / dim))  # (dim/2,)
        
        # Compute ratio r(d) = L / λ_d
        r = self.original_max_seq_len / wavelengths  # (dim/2,)
        
        # Compute ramp function γ(r)
        gamma = self._compute_ramp_function(r)  # (dim/2,)
        
        # NTK-by-parts interpolation:
        # θ'_d = (1 - γ(r)) * θ_d/s + γ(r) * θ_d
        base_thetas = 1.0 / (self.base ** (d_indices / dim))  # (dim/2,)
        interpolated_thetas = (1 - gamma) * (base_thetas / scale_factor) + gamma * base_thetas
        
        # Expand to full dimension (interleave for cos/sin pairs)
        thetas = (
            interpolated_thetas.unsqueeze(1)  # (dim/2, 1)
            .expand(-1, 2)  # (dim/2, 2)
            .reshape(-1)  # (dim,)
        )
        
        # Compute position indices
        pos = torch.arange(0, max_seq_len, dtype=dtype, device=device)  # (max_seq_len,)
        
        # Compute rotations
        rotate = torch.outer(pos, thetas)  # (max_seq_len, dim)
        
        # Apply attention scaling
        rotate = rotate * self.attention_scale
        
        return torch.cos(rotate), torch.sin(rotate)

    def _update_cache(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        scale_factor: float,
    ) -> None:
        """Update the cos/sin cache if necessary."""
        needs_rebuild = (
            self.cos.numel() == 0  # initial build
            or seq_len > self.max_seq_len
            or self.cos.device != device
            or self.cos.dtype != dtype
        )
        if not needs_rebuild:
            return

        target_len = max(seq_len, self.max_seq_len)
        _cos, _sin = self._compute_rotates(
            max_seq_len=target_len,
            dim=self.dim,
            device=device,
            dtype=dtype,
            scale_factor=scale_factor,
        )
        self.cos = _cos
        self.sin = _sin
        self.max_seq_len = target_len

    def forward(
        self, x: Float[Tensor, "... seq_len {self.dim}"]
    ) -> Float[Tensor, "... seq_len {self.dim}"]:  # noqa: F821
        """
        Apply YaRN rotary positional encoding to input tensor.
        """
        seq_len = x.size(-2)
        
        # Dynamic scaling: adjust scale_factor based on current sequence length
        if self.enable_dynamic_scaling:
            dynamic_scale = max(1.0, seq_len / self.original_max_seq_len)
            # Recompute attention scale for dynamic scaling
            self.attention_scale = 0.1 * math.log(dynamic_scale) + 1.0
            
            # Update cache with dynamic scale if needed
            if dynamic_scale != self.scale_factor:
                self._update_cache(
                    seq_len=seq_len,
                    device=x.device,
                    dtype=x.dtype,
                    scale_factor=dynamic_scale,
                )
        else:
            self._update_cache(
                seq_len=seq_len,
                device=x.device,
                dtype=x.dtype,
                scale_factor=self.scale_factor,
            )

        cos = self.cos[:seq_len, :]  # (seq_len, dim)
        sin = self.sin[:seq_len, :]  # (seq_len, dim)

        return (x * cos) + (rotate_half(x) * sin)
