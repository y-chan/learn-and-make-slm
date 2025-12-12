import torch
from utils.mask import make_pad_mask, make_non_pad_mask


def test_make_pad_mask_basic():
    lengths = [5, 3, 2]

    mask = make_pad_mask(lengths)

    expected = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
            [False, False, True, True, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_make_pad_mask_with_long_tensor():
    lengths = torch.LongTensor([4, 2, 3])

    mask = make_pad_mask(lengths)

    expected = torch.tensor(
        [
            [False, False, False, False],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_make_pad_mask_with_reference_tensor():
    lengths = [3, 2]
    xs = torch.zeros((2, 2, 4))

    mask = make_pad_mask(lengths, xs)

    assert mask.shape == xs.shape
    expected = torch.tensor(
        [
            [[False, False, False, True], [False, False, False, True]],
            [[False, False, True, True], [False, False, True, True]],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_make_pad_mask_with_length_dim():
    lengths = [3, 2]
    xs = torch.zeros((2, 4, 4))

    mask = make_pad_mask(lengths, xs, length_dim=1)

    assert mask.shape == xs.shape
    # Check that dim 1 is the length dimension
    # For lengths [3, 2], the mask should have True values at indices >= length
    expected = torch.tensor(
        [
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ],
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_make_pad_mask_with_maxlen():
    lengths = [3, 2]

    mask = make_pad_mask(lengths, maxlen=5)

    expected = torch.tensor(
        [
            [False, False, False, True, True],
            [False, False, True, True, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_make_non_pad_mask_basic():
    lengths = [5, 3, 2]

    mask = make_non_pad_mask(lengths)

    expected = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_make_non_pad_mask_is_inverse_of_pad_mask():
    lengths = [4, 2, 3, 1]

    pad_mask = make_pad_mask(lengths)
    non_pad_mask = make_non_pad_mask(lengths)

    torch.testing.assert_close(~pad_mask, non_pad_mask)


def test_make_non_pad_mask_with_reference_tensor():
    lengths = [3, 2]
    xs = torch.zeros((2, 2, 4))

    mask = make_non_pad_mask(lengths, xs)

    assert mask.shape == xs.shape
    expected = torch.tensor(
        [
            [[True, True, True, False], [True, True, True, False]],
            [[True, True, False, False], [True, True, False, False]],
        ]
    )
    torch.testing.assert_close(mask, expected)
