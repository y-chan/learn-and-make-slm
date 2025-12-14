from torch import Tensor, device as TorchDevice


def to_device(data: dict, device: TorchDevice):
    """Move data to device."""
    new_data = {}
    for k, v in data.items():
        if isinstance(v, Tensor):
            new_data[k] = v.to(device, non_blocking=True)
        else:
            new_data[k] = v

    return new_data
