from torch import nn, Tensor

from utils.randn import nonzero_randn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # self.one_hot_vec = torch.eye(num_embeddings)
        # NOTE: Linearモジュールのコメントを参照
        self.w = nn.Parameter(nonzero_randn(num_embeddings, embedding_dim) * (embedding_dim**-0.5))

    def forward(self, x: Tensor) -> Tensor:
        # one_hot = self.one_hot_vec[x]
        # return one_hot @ self.w
        # torch.eyeでone hot vectorを作るのはOut of Memoryを引き起こすが、
        # 実は単に重みのindexを指定するだけで良く、これならOOMを回避できる
        return self.w[x]
