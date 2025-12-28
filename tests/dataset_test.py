import numpy as np
import torch
from dataset import dataset_collate


def test_dataset_collate_basic():
    batch = [
        {"story": [1, 2, 3, 4, 5]},
        {"story": [6, 7, 8]},
        {"story": [9, 10]},
    ]

    result = dataset_collate(batch)

    assert result["tokens_ids"].shape == (3, 5)
    assert result["lengths"].shape == (3,)
    assert isinstance(result["tokens_ids"], torch.Tensor)
    assert isinstance(result["lengths"], torch.Tensor)
    assert result["tokens_ids"].dtype == torch.int64
    assert result["lengths"].dtype == torch.int64


def test_dataset_collate_padding():
    batch = [
        {"story": [1, 2, 3]},
        {"story": [4, 5]},
    ]

    result = dataset_collate(batch)

    expected_tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64)
    expected_lengths = torch.tensor([3, 2])
    torch.testing.assert_close(result["tokens_ids"], expected_tokens)
    torch.testing.assert_close(result["lengths"], expected_lengths)


def test_dataset_collate_torch_convert_false():
    batch = [{"story": [1, 2, 3]}]

    result = dataset_collate(batch, torch_convert=False)

    assert isinstance(result["tokens_ids"], np.ndarray)
    assert isinstance(result["lengths"], np.ndarray)


def test_dataset_collate_max_length_truncation():
    np.random.seed(42)
    batch = [{"story": list(range(100))}]

    result = dataset_collate(batch, max_length=10)

    assert result["tokens_ids"].shape == (1, 10)
    assert result["lengths"].item() == 10
    # 切り詰め後も連続した系列である
    tokens = result["tokens_ids"][0].tolist()
    for i in range(1, len(tokens)):
        assert tokens[i] == tokens[i - 1] + 1
