from torch import SymInt, Tensor, randn, where


def nonzero_randn(*shape: int | SymInt, epsilon: float = 1e-6) -> Tensor:
    """
    Generates a random Tensor with no zero elements.

    Parameters
    ----------
    shape : tuple[int, SymInt]

    epsilon : float
        A minimum absolute value for the elements.

    Returns
    -------
    Tensor
        Generated random Tensor.
    """
    if epsilon <= 0:
        msg = f"epsilon must be positive, but got {epsilon}"
        raise ValueError(msg)

    x = randn(*shape)
    mask = x.abs() < epsilon
    return where(mask, epsilon * x.sign(), x)
