from torch import SymInt, Tensor, nn


def nonzero_randn(*shape: int | SymInt, epsilon: float = 1e-6) -> Tensor:
    """
    Generates a random Tensor with no zero elements.

    Parameters
    ----------
    shape : tuple[int, SymInt]

    epsilon : float
        A minimum absolute value for the elements.

    Returns
    -------
    Tensor
        Generated random Tensor.
    """
    if epsilon <= 0:
        msg = f"epsilon must be positive, but got {epsilon}"
        raise ValueError(msg)

    x = torch.randn(shape)
    mask = x.abs() < epsilon
    return torch.where(mask, epsilon * x.sign(), x)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            入力 Tensor の次元数
        out_features : int
            出力 Tensor の次元数
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # NOTE: 一旦適当な行列で初期化しておくと、iteration ごとにいい感じの Tensor に収束していくので
        # 具体的な中身の Tensor を気にする必要はない。ただし要素に 0 があると影響が出る可能性があるので避けたい。
        self.weight = nn.Parameter(nonzero_randn(out_features, in_features))
        self.bias = nn.Parameter(nonzero_randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        # y = w * x + b
        return x @ self.weight.t() + self.bias.t()


if __name__ == "__main__":
    import torch

    linear = Linear(10, 20)

    x = torch.randn(2, 10)
    assert linear(x).shape == (2, 20)
    print("Linear test passed")
