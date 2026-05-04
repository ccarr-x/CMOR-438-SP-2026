from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[Any]

class PCA:
    """Principal Component Analysis for dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=None
        Number of principal components to keep. If None, all components are kept.
    use_svd : bool, default=False
        Whether to use an SVD-based PCA implementation instead of covariance matrix eigen decomposition.
    """

    def __init__(self, n_components: Union[int, None] = None, use_svd: bool = False) -> None:
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be at least 1 or None.")
        if not isinstance(use_svd, bool):
            raise ValueError("use_svd must be a boolean.")
        self.n_components = n_components
        self.use_svd = use_svd
        self.components_: _Array = np.empty((0, 0))
        self.explained_variance_: _Array = np.empty(0)
        self.explained_variance_ratio_: _Array = np.empty(0)
        self.mean_: _Array = np.empty(0)
        self.n_features_: int = 0
        self.mean_: _Array = np.empty(0)
        self.n_features_: int = 0

    def fit(self, X: _Array) -> PCA:
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCA
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        n_samples, n_features = X_arr.shape

        # Center the data
        mean = np.mean(X_arr, axis=0)
        X_centered = X_arr - mean

        if self.use_svd:
            U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
            n_components = self.n_components
            if n_components is None:
                n_components = min(n_samples, n_features)
            if n_components > min(n_samples, n_features):
                raise ValueError("n_components cannot be greater than min(n_samples, n_features).")
            components = VT[:n_components]
            explained_variance = (S[:n_components] ** 2) / (n_samples - 1)
        else:
            cov_matrix = np.cov(X_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues_sorted = eigenvalues[sorted_idx]
            eigenvectors_sorted = eigenvectors[:, sorted_idx]
            if self.n_components is not None:
                if self.n_components > min(n_samples, n_features):
                    raise ValueError("n_components cannot be greater than min(n_samples, n_features).")
                eigenvalues_sorted = eigenvalues_sorted[: self.n_components]
                eigenvectors_sorted = eigenvectors_sorted[:, : self.n_components]
            components = eigenvectors_sorted.T
            explained_variance = eigenvalues_sorted

        total_variance = np.sum(np.var(X_centered, axis=0, ddof=1))
        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / total_variance
        self.mean_ = mean
        self.n_features_ = n_features
        return self
    
    def transform(self, X: _Array) -> _Array:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns -------
        X_transformed : array of shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_.size == 0:
            raise ValueError("Model is not fitted. Call fit(X) before transform(X).")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but PCA was fitted with {self.n_features_}."
            )
        X_centered = X_arr - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: _Array) -> _Array:
        """Fit the model with X and apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: _Array) -> _Array:
        """Transform data back to its original space.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_components)
            Data in the reduced space.

        Returns
        -------
        X_approx : array of shape (n_samples, n_features)
            Approximation of the original data.
        """
        if self.components_.size == 0:
            raise ValueError("Model is not fitted. Call fit(X) before inverse_transform(X).")
        X_transformed_arr = np.asarray(X_transformed, dtype=float)
        if X_transformed_arr.ndim != 2:
            raise ValueError("X_transformed must be two-dimensional (n_samples, n_components).")
        if X_transformed_arr.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"X_transformed has {X_transformed_arr.shape[1]} components, but PCA was fitted with {self.components_.shape[0]}."
            )
        return X_transformed_arr @ self.components_ + self.mean_
