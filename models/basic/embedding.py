from torch import nn, Tensor

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    import torch

    embedding = Embedding(50, 20)

    x = torch.randint(0, 50, (2, 10))
    assert embedding(x).shape == (2, 10, 20)
    print("Embedding test passed")
