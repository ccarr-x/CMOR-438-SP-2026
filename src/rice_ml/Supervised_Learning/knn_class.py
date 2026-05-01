"""k-nearest neighbors model for classification or regression."""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[Any]
TaskType = Literal["classification", "regression"]


class KNearestNeighbors:
    """Simple k-nearest neighbors using Euclidean distance.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest training samples used for each prediction.
    task : {"classification", "regression"}, default="classification"
        Prediction mode used for labeling can be specified.
    """

    def __init__(self, n_neighbors: int = 5, task: TaskType = "classification") -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        self.n_neighbors = int(n_neighbors)
        self.task = task

    def fit(self, X: _Array, y: _Array) -> KNearestNeighbors:
        """Store training data for neighbor queries."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError("n_neighbors cannot be greater than number of training samples.")

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        return self

    def _euclidean_distances(self, x: _Array) -> _Array:
        """Compute Euclidean distance from one sample to all training points."""
        diff = self.X_train_ - x
        return np.sqrt(np.sum(diff * diff, axis=1))

    def _predict_classification(self, x: _Array) -> Union[int, float, str]:
        distances = self._euclidean_distances(x)
        neighbor_idx = np.argsort(distances)[: self.n_neighbors]
        neighbor_labels = self.y_train_[neighbor_idx]

        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def _predict_regression(self, x: _Array) -> float:
        distances = self._euclidean_distances(x)
        neighbor_idx = np.argsort(distances)[: self.n_neighbors]
        neighbor_values = self.y_train_[neighbor_idx].astype(float)
        return float(np.mean(neighbor_values))

    def predict(self, X: _Array) -> _Array:
        """Predict class labels or regression values for samples in X."""
        if not hasattr(self, "X_train_"):
            raise ValueError("Model is not fitted. Call fit(X, y) before predict(X).")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError("X must be one- or two-dimensional.")
        if X_arr.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X feature count must match training data feature count.")

        if self.task == "classification":
            preds = [self._predict_classification(x) for x in X_arr]
            return np.asarray(preds, dtype=self.y_train_.dtype)

        preds = [self._predict_regression(x) for x in X_arr]
        return np.asarray(preds, dtype=float)
