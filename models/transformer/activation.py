import torch

from models.basic.linear import Linear


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y = 1.0 / (1.0 + (-x).exp())
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # 基本的には自動微分に任せるが、ここだけはこの方が実装が楽なので実装
        y = ctx.saved_tensors[0]
        return grad_output * y * (1.0 - y)

    @staticmethod
    def symbolic(g, x: torch.Tensor):
        # ONNX export時にこのメソッドが自動的に呼ばれる
        return g.op("Sigmoid", x)


sigmoid = SigmoidFunction.apply


class Swish(torch.nn.Module):
    def __init__(self, beta: float = 1.0, beta_is_learnable: bool = False):
        """
        Implements Swish.

        See: https://arxiv.org/abs/1710.05941

        Parameters
        ----------
        beta : float, optional
            Sharpness parameter, by default 1.0
        mark_beta_learnable : bool, optional
            Whether beta is learnable, by default False

            Why default to False?
            In the original paper, it was shown experimentally that even when beta is marked as a learnable parameter, it converges to a value close to 1.0, indicating that fixing it to 1.0 in practice causes no issues.
        """
        super().__init__()
        if beta_is_learnable:
            self.beta = torch.nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * sigmoid(self.beta * x)


class GLU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ):
        super().__init__()
        self.proj = Linear(dim_in, dim_hidden * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, v = self.proj(x).chunk(2, dim=-1)
        return w * sigmoid(v)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ):
        super().__init__()
        self.proj = Linear(dim_in, dim_hidden * 2)
        self._swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, v = self.proj(x).chunk(2, dim=-1)
        return w * self._swish(v)
