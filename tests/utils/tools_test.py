import torch
from utils.tools import to_device


def test_to_device_moves_tensors():
    device = torch.device("cpu")
    data = {
        "tensor1": torch.randn(2, 3),
        "tensor2": torch.randn(4, 5),
    }

    result = to_device(data, device)

    assert result["tensor1"].device == device
    assert result["tensor2"].device == device


def test_to_device_moves_long_tensors():
    device = torch.device("cpu")
    data = {
        "long_tensor": torch.LongTensor([1, 2, 3]),
    }

    result = to_device(data, device)

    assert result["long_tensor"].device == device
    assert result["long_tensor"].dtype == torch.long


def test_to_device_preserves_non_tensor_values():
    device = torch.device("cpu")
    data = {
        "tensor": torch.randn(2, 3),
        "string": "hello",
        "number": 42,
        "list": [1, 2, 3],
    }

    result = to_device(data, device)

    assert result["string"] == "hello"
    assert result["number"] == 42
    assert result["list"] == [1, 2, 3]


def test_to_device_returns_same_dict():
    device = torch.device("cpu")
    data = {
        "tensor": torch.randn(2, 3),
    }

    result = to_device(data, device)

    assert result is data
