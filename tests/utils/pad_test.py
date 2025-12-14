import numpy as np
from utils.pad import pad_1D


def test_pad_1d_basic():
    inputs = [np.array([1, 2, 3]), np.array([4, 5])]

    result = pad_1D(inputs)

    expected = np.array([[1, 2, 3], [4, 5, 0]])
    np.testing.assert_array_equal(result, expected)


def test_pad_1d_custom_pad_value():
    inputs = [np.array([1, 2]), np.array([3, 4, 5, 6])]

    result = pad_1D(inputs, pad=-1.0)

    expected = np.array([[1, 2, -1, -1], [3, 4, 5, 6]])
    np.testing.assert_array_equal(result, expected)


def test_pad_1d_single_input():
    inputs = [np.array([1, 2, 3, 4])]

    result = pad_1D(inputs)

    expected = np.array([[1, 2, 3, 4]])
    np.testing.assert_array_equal(result, expected)


def test_pad_1d_same_length():
    inputs = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

    result = pad_1D(inputs)

    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_array_equal(result, expected)


def test_pad_1d_output_shape():
    inputs = [np.array([1]), np.array([2, 3]), np.array([4, 5, 6, 7, 8])]

    result = pad_1D(inputs)

    assert result.shape == (3, 5)
