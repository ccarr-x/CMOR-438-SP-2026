"""Preprocessing utilities for splitting and standardizing data."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list]


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split feature/target data into train and test sets.

    Parameters
    ----------
    X : array-like
        Feature matrix with shape (n_samples, n_features).
    y : array-like
        Target array with shape (n_samples,).
    test_size : float, default=0.2
        Fraction of samples assigned to the test set. Must satisfy 0 < test_size < 1.
    random_state : int or None, default=None
        Optional random seed for reproducibility.
    shuffle : bool, default=True
        If True, shuffle indices before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of np.ndarray
        Arrays containing train and test splits.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1.")
    if X_arr.shape[0] < 2:
        raise ValueError("At least 2 samples are required to split.")

    n_samples = X_arr.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    n_test = min(max(n_test, 1), n_samples - 1)

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def standardize(
    X_train: ArrayLike,
    X_test: Optional[ArrayLike] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Standardize features using training-set statistics.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    X_test : array-like or None, default=None
        Optional test feature matrix transformed using X_train mean/std.

    Returns
    -------
    np.ndarray or tuple(np.ndarray, np.ndarray)
        Standardized training data, and standardized test data if X_test is provided.
    """
    X_train_arr = np.asarray(X_train, dtype=float)
    mean = X_train_arr.mean(axis=0)
    std = X_train_arr.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train_arr - mean) / std_safe

    if X_test is None:
        return X_train_scaled

    X_test_arr = np.asarray(X_test, dtype=float)
    X_test_scaled = (X_test_arr - mean) / std_safe
    return X_train_scaled, X_test_scaled


def normalize(X: ArrayLike) -> np.ndarray:
    """
    Normalize features to unit norm.

    Parameters
    ----------
    X : array-like
        Feature matrix to normalize.

    Returns
    -------
    np.ndarray
        Normalized feature matrix.
    """
    X_arr = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
    norms_safe = np.where(norms == 0, 1.0, norms)
    return X_arr / norms_safe


def min_max_scale(X: ArrayLike) -> np.ndarray:
    """
    Scale features to the [0, 1] range.

    Parameters
    ----------
    X : array-like
        Feature matrix to scale.

    Returns
    -------
    np.ndarray
        Scaled feature matrix.
    """
    X_arr = np.asarray(X, dtype=float)
    min_vals = X_arr.min(axis=0)
    max_vals = X_arr.max(axis=0)
    ranges = max_vals - min_vals
    ranges_safe = np.where(ranges == 0, 1.0, ranges)
    return (X_arr - min_vals) / ranges_safe
    

def encode_labels(y: ArrayLike) -> np.ndarray:
    """
    Encode categorical labels as integers.

    Parameters
    ----------
    y : array-like
        Target labels to encode.

    Returns
    -------
    np.ndarray
        Encoded integer labels.
    """
    y_arr = np.asarray(y)
    unique_labels, encoded = np.unique(y_arr, return_inverse=True)
    return encoded


def transform_labels(y: ArrayLike, mapping: dict) -> np.ndarray:
    """
    Transform labels using a provided mapping.

    Parameters
    ----------
    y : array-like
        Target labels to transform.
    mapping : dict
        Dictionary mapping original labels to new values.

    Returns
    -------
    np.ndarray
        Transformed labels.
    """
    y_arr = np.asarray(y)
    transformed = np.vectorize(mapping.get)(y_arr)
    if np.any(transformed == None):
        raise ValueError("All labels in y must be present in the mapping keys.")
    return transformed