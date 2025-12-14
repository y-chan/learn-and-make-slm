from torch import Tensor, nn

from utils.randn import nonzero_randn


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
        # また、in_featuresで割ることで、重みのスケールを安定させる。
        self.weight = nn.Parameter(nonzero_randn(out_features, in_features) * (in_features**-0.5))
        self.bias = nn.Parameter(nonzero_randn(out_features) * (out_features**-0.5))

    def forward(self, x: Tensor) -> Tensor:
        # y = w * x + b
        return x @ self.weight.T + self.bias
