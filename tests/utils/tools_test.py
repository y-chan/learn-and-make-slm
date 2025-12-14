import pytest
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


def test_to_device_any_tensor_type_is_converted():
    device = torch.device("cpu")
    long_tensor = torch.LongTensor([1, 2, 3])
    float_tensor = torch.FloatTensor([1.0, 2.0, 3.0])
    data = {
        "long": long_tensor,
        "float": float_tensor,
    }

    result = to_device(data, device)

    assert result["long"].device == device
    assert result["float"].device == device
    assert result["long"].dtype == torch.long
    assert result["float"].dtype == torch.float


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


def test_to_device_returns_new_dict():
    device = torch.device("cpu")
    data = {
        "tensor": torch.randn(2, 3),
    }

    result = to_device(data, device)

    assert result is not data


# GPU tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_device_moves_to_cuda():
    device = torch.device("cuda:0")
    data = {
        "tensor": torch.randn(2, 3),
    }

    result = to_device(data, device)

    assert result["tensor"].device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_device_moves_long_tensor_to_cuda():
    device = torch.device("cuda:0")
    data = {
        "long_tensor": torch.LongTensor([1, 2, 3]),
    }

    result = to_device(data, device)

    assert result["long_tensor"].device == device
    assert result["long_tensor"].dtype == torch.long


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_device_moves_mixed_tensors_to_cuda():
    device = torch.device("cuda:0")
    data = {
        "float": torch.FloatTensor([1.0, 2.0]),
        "long": torch.LongTensor([1, 2, 3]),
        "int": torch.IntTensor([4, 5]),
        "double": torch.DoubleTensor([6.0, 7.0]),
    }

    result = to_device(data, device)

    for key in ["float", "long", "int", "double"]:
        assert result[key].device == device

    assert result["float"].dtype == torch.float
    assert result["long"].dtype == torch.long
    assert result["int"].dtype == torch.int
    assert result["double"].dtype == torch.double
