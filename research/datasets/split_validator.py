"""
Pure-NumPy validator for official NYU Depth V2 `splits.mat` 1-based MATLAB indices.
Separated from PyTorch imports so both training and lightweight CI integrity tests
execute the exact same validation logic.
"""

from typing import List, Tuple
import numpy as np


def validate_nyu_split_indices(
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    total: int,
) -> Tuple[List[int], List[int]]:
    """
    Validates 1-based MATLAB train/test split indices against dataset size `total`.
    Returns 0-based `(train_idx, test_idx)` or raises ValueError / IndexError.
    """
    flat_train = np.asarray(raw_train).reshape(-1)
    flat_test = np.asarray(raw_test).reshape(-1)

    if flat_train.size == 0 or flat_test.size == 0:
        raise ValueError("Official splits.mat train/test split cannot be empty!")

    if not (
        np.all(np.isfinite(flat_train))
        and np.all(np.isfinite(flat_test))
        and np.all(np.mod(flat_train, 1) == 0)
        and np.all(np.mod(flat_test, 1) == 0)
    ):
        raise ValueError("Official splits.mat contains non-integer or non-finite indices!")

    train_idx = [int(i) - 1 for i in flat_train]
    test_idx = [int(i) - 1 for i in flat_test]

    if len(train_idx) != len(set(train_idx)) or len(test_idx) != len(set(test_idx)):
        raise ValueError("Official splits.mat contains duplicate indices within a split!")

    if set(train_idx) & set(test_idx):
        raise ValueError("Official splits.mat contains overlapping train/test indices!")

    if any(i < 0 or i >= total for i in train_idx) or any(i < 0 or i >= total for i in test_idx):
        raise IndexError(f"splits.mat index out of bounds for dataset size {total}")

    return train_idx, test_idx
