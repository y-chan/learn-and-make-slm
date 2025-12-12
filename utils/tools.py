from torch import Tensor, LongTensor, device as TorchDevice


def to_device(data: dict, device: TorchDevice):
    """Move data to device."""
    for k, v in data.items():
        if isinstance(v, Tensor):
            data[k] = v.to(device, non_blocking=True)

    return data
