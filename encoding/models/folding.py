import numpy as np
import random
from typing import List, Tuple, Optional
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
import logging


def create_folds(
    n_samples: int,
    fold_type: str,
    n_folds: int,
    chunk_length: Optional[int] = None,
    trim_size: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Create train/test indices according to the specified folding strategy

    Args:
        n_samples: Number of samples in the dataset
        fold_type: Type of folding - 'chunked', 'chunked_trimmed', 'kfold',
                  'kfold_trimmed', 'chunked_contiguous', 'timeseries', 'group'
        n_folds: Number of folds to create
        chunk_length: Length of chunks for chunked folding types
        trim_size: Number of TRs to trim from beginning and end of test folds/chunks
                  (used for 'chunked_trimmed' and 'kfold_trimmed' fold types)
        groups: Group labels for group-based folding

    Returns:
        List of (train_indices, test_indices) tuples
    """
    print(f"THIS IS THE FOLDING TYPE: {fold_type} inside folding.py")
    if fold_type == "chunked":
        # Current chunked implementation with random shuffling
        return create_chunked_folds(n_samples, n_folds, chunk_length, shuffle=True)
    elif fold_type == "chunked_trimmed":
        # Chunked with trimmed test chunks to reduce autocorrelation
        if trim_size is None:
            trim_size = 5  # Default trim size of 5 TRs
        return create_chunked_folds_trimmed(
            n_samples, n_folds, chunk_length, trim_size, shuffle=True
        )
    elif fold_type == "chunked_contiguous":
        # Chunked but maintaining original order
        return create_chunked_folds(n_samples, n_folds, chunk_length, shuffle=False)
    elif fold_type == "kfold":
        # Regular KFold with shuffle=False for contiguity
        kf = KFold(n_splits=n_folds, shuffle=False)
        return list(kf.split(range(n_samples)))
    elif fold_type == "kfold_trimmed":
        # KFold with trimmed test folds to reduce autocorrelation
        if trim_size is None:
            trim_size = 5  # Default trim size of 5 TRs
        return create_kfold_trimmed(n_samples, n_folds, trim_size)
    elif fold_type == "timeseries":
        # Use TimeSeriesSplit for temporal forward-looking validation
        tscv = TimeSeriesSplit(n_splits=n_folds)
        return list(tscv.split(range(n_samples)))
    elif fold_type == "group":
        if groups is None:
            raise ValueError("Groups must be provided for group folding")
        gkf = GroupKFold(n_splits=n_folds)
        return list(gkf.split(range(n_samples), groups=groups))
    else:
        raise ValueError(f"Unknown folding type: {fold_type}")


def create_chunked_folds(
    n_samples: int, n_folds: int, chunk_length: int, shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    """Create KFold splits that respect chunk structure of fMRI data.

    Args:
        n_samples: Number of samples in the dataset
        n_folds: Number of folds to create
        chunk_length: Length of chunks
        shuffle: Whether to shuffle chunks or keep them in original order

    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Calculate how many complete chunks we can have
    n_complete_chunks = n_samples // chunk_length
    chunk_indices = list(range(n_complete_chunks))

    if shuffle:
        random.shuffle(chunk_indices)

    # Divide chunks evenly into n_folds
    chunks_per_fold = n_complete_chunks // n_folds
    if chunks_per_fold == 0:
        # Not enough chunks for the requested folds, fall back to regular KFold
        logging.warning(
            "Not enough chunks for the requested folds, falling back to regular KFold"
        )
        kf = KFold(n_splits=n_folds, shuffle=shuffle)
        return list(kf.split(range(n_samples)))

    # Create exactly n_folds splits
    splits = []
    for i in range(n_folds):
        # Select test chunks for this fold
        start_idx = i * chunks_per_fold
        end_idx = (i + 1) * chunks_per_fold if i < n_folds - 1 else n_complete_chunks
        test_chunks = chunk_indices[start_idx:end_idx]

        # All other chunks are for training
        train_chunks = [c for c in chunk_indices if c not in test_chunks]

        # Convert chunk indices to sample indices
        test_indices = []
        for chunk in test_chunks:
            start = chunk * chunk_length
            end = start + chunk_length
            test_indices.extend(range(start, min(end, n_samples)))

        train_indices = []
        for chunk in train_chunks:
            start = chunk * chunk_length
            end = start + chunk_length
            train_indices.extend(range(start, min(end, n_samples)))

        splits.append((train_indices, test_indices))

    return splits


def create_chunked_folds_trimmed(
    n_samples: int,
    n_folds: int,
    chunk_length: int,
    trim_size: int = 5,
    shuffle: bool = True,
) -> List[Tuple[List[int], List[int]]]:
    """Create KFold splits with trimmed test chunks to reduce autocorrelation effects.

    This approach reduces the impact of BOLD signal autocorrelation by removing the first
    and last [trim_size] TRs from each test chunk. This eliminates the time points most
    affected by autocorrelation at chunk boundaries, resulting in more independent test sets.
    The full chunks are still used for training data.

    Args:
        n_samples: Number of samples in the dataset
        n_folds: Number of folds to create
        chunk_length: Length of chunks
        trim_size: Number of time points to trim from beginning and end of test chunks
        shuffle: Whether to shuffle chunks or keep them in original order

    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Calculate how many complete chunks we can have
    n_complete_chunks = n_samples // chunk_length
    chunk_indices = list(range(n_complete_chunks))

    if shuffle:
        random.shuffle(chunk_indices)

    # Divide chunks evenly into n_folds
    chunks_per_fold = n_complete_chunks // n_folds
    if chunks_per_fold == 0:
        logging.warning(
            "Not enough chunks for the requested folds, falling back to regular KFold"
        )
        kf = KFold(n_splits=n_folds, shuffle=False)
        return list(kf.split(range(n_samples)))

    # Create exactly n_folds splits
    splits = []
    logging.info(f"Creating {n_folds} folds with {chunks_per_fold} chunks per fold")
    for i in range(n_folds):
        # Select test chunks for this fold
        start_idx = i * chunks_per_fold
        end_idx = (i + 1) * chunks_per_fold if i < n_folds - 1 else n_complete_chunks
        test_chunks = chunk_indices[start_idx:end_idx]

        # All other chunks are for training
        train_chunks = [c for c in chunk_indices if c not in test_chunks]

        # Convert chunk indices to sample indices
        test_indices = []
        for chunk in test_chunks:
            # Full chunk range
            chunk_start = chunk * chunk_length
            chunk_end = min(chunk_start + chunk_length, n_samples)

            # Trim the boundaries to reduce autocorrelation
            trimmed_start = chunk_start + trim_size
            trimmed_end = chunk_end - trim_size

            # Only add if we have valid indices after trimming
            if trimmed_start < trimmed_end:
                test_indices.extend(range(trimmed_start, trimmed_end))

        # For training, use the full chunks
        train_indices = []
        for chunk in train_chunks:
            start = chunk * chunk_length
            end = min(start + chunk_length, n_samples)
            train_indices.extend(range(start, end))

        splits.append((train_indices, test_indices))

    return splits


def create_kfold_trimmed(
    n_samples: int,
    n_folds: int,
    trim_size: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """Create KFold splits with trimmed test folds to reduce autocorrelation effects.

    This approach reduces the impact of temporal autocorrelation by removing the first
    and last [trim_size] samples from each test fold. This is useful for temporal data
    like fMRI where adjacent time points are correlated, and you want to reduce the
    dependency between training and test sets at fold boundaries.

    Args:
        n_samples: Number of samples in the dataset
        n_folds: Number of folds to create
        trim_size: Number of time points to trim from beginning and end of test folds

    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Use regular KFold to get the base splits
    kf = KFold(n_splits=n_folds, shuffle=False)  # ALWAYS SHUFFLE=FALSE
    base_splits = list(kf.split(range(n_samples)))

    # Apply trimming to test sets
    trimmed_splits = []
    logging.info(
        f"Creating {n_folds} KFold splits with {trim_size} sample trim on test folds"
    )

    for train_indices, test_indices in base_splits:
        # Convert to lists for easier manipulation
        train_indices = list(train_indices)
        test_indices = list(test_indices)

        # Trim the test set - remove first and last trim_size samples
        if len(test_indices) > 2 * trim_size:
            trimmed_test = test_indices[trim_size:-trim_size]
        else:
            # If test fold is too small to trim, keep original
            logging.warning(
                f"Test fold too small ({len(test_indices)} samples) to trim {trim_size} "
                f"from each end, keeping original test set"
            )
            trimmed_test = test_indices

        # Training set remains unchanged
        trimmed_splits.append((train_indices, trimmed_test))

    return trimmed_splits
