"""Time-series cross-validation utilities with purging to reduce leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple


@dataclass
class PurgedSplit:
    train_idx: Sequence[int]
    test_idx: Sequence[int]


def rolling_purged_splits(n: int, n_splits: int = 5, purge: int = 0, min_train: int = 50) -> Iterator[PurgedSplit]:
    """Generate rolling time-ordered splits.

    Args:
        n: total number of samples (assumed time-ordered)
        n_splits: number of evaluation slices
        purge: number of samples to drop between train and test (gap)
        min_train: minimum training size to yield a split
    """
    if n <= min_train or n_splits <= 0:
        return iter(())
    test_size = max(1, (n - min_train) // n_splits)
    for split in range(n_splits):
        test_start = min_train + split * test_size
        test_end = min(test_start + test_size, n)
        train_end = max(min_train, test_start - purge)
        if train_end - 0 < min_train:
            continue
        yield PurgedSplit(train_idx=tuple(range(0, train_end)), test_idx=tuple(range(test_start, test_end)))
