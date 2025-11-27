import torch

from models.basic.embedding import Embedding


def test_embedding_gathers_rows():
    embedding = Embedding(50, 20)
    x = torch.randint(0, 50, (2, 10))

    y = embedding(x)

    assert y.shape == (2, 10, 20)
