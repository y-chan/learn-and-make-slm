import numpy as np


def pad_1D(inputs: np.ndarray | list[np.ndarray], pad: float = 0) -> np.ndarray:
    def pad_data(x: np.ndarray, length: int) -> np.ndarray:
        x_padded = np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=pad)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len) for x in inputs])

    return padded
