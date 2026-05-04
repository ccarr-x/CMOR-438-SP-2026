from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[Any]

class KMeans:

    """K-means clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int, optional
        Seed for reproducible initialization of cluster centers.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Union[int, None] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: _Array = np.empty((0, 0))
        self.labels_: _Array = np.empty(0, dtype=int)
        self.inertia_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: _Array) -> KMeans:
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        n_samples, n_features = X_arr.shape
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n_samples})."
            )

        # Initialize cluster centers randomly from the data points
        rng = np.random.default_rng(self.random_state)
        initial_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.cluster_centers_ = X_arr[initial_idx]

        for iteration in range(self.max_iter):
            # Assign labels based on closest center
            distances = np.linalg.norm(X_arr[:, np.newaxis] - self.cluster_centers_, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # Compute inertia (sum of squared distances to closest center)
            new_inertia = np.sum((X_arr - self.cluster_centers_[new_labels]) ** 2)

            # Check for convergence
            if iteration > 0 and abs(self.inertia_ - new_inertia) <= self.tol * self.inertia_:
                break

            self.labels_ = new_labels
            self.inertia_ = new_inertia

            # Update cluster centers
            for i in range(self.n_clusters):
                cluster_points = X_arr[self.labels_ == i]
                if cluster_points.size:
                    self.cluster_centers_[i] = cluster_points.mean(axis=0)
                else:
                    # Reinitialize empty clusters with a random sample
                    self.cluster_centers_[i] = X_arr[rng.integers(n_samples)]

        self.n_iter_ = iteration + 1
        return self
    
    def predict(self, X: _Array) -> _Array:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        if self.cluster_centers_.shape[0] == 0:
            raise ValueError("KMeans instance is not fitted yet. Call 'fit' with appropriate data.")

        distances = np.linalg.norm(X_arr[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: _Array) -> _Array:
        """Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_
    
    def fit_transform(self, X: _Array) -> _Array:
        """Compute cluster centers and transform X to cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            Transformed data in cluster-distance space.
        """
        self.fit(X)
        X_arr = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X_arr[:, np.newaxis] - self.cluster_centers_, axis=2)
        return distances
    