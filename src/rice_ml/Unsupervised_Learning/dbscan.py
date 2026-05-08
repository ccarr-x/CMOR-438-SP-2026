from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[Any]

class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: _Array = np.empty(0, dtype=int)
        self.n_clusters_: int = 0

    def fit(self, X: _Array) -> DBSCAN:
        """Perform DBSCAN clustering from features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        self : DBSCAN
            Fitted estimator.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        n_samples = X_arr.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)
        
        # Compute pairwise distances
        distances = np.linalg.norm(X_arr[:, np.newaxis] - X_arr, axis=2)
        
        # Find neighbors for each point
        neighbors = [np.where(distances[i] <= self.eps)[0] for i in range(n_samples)]
        
        # Identify core points
        core_points = np.where(np.array([len(n) for n in neighbors]) >= self.min_samples)[0]
        
        # Cluster assignment
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        for point in core_points:
            if visited[point]:
                continue
            
            # BFS to expand cluster
            queue = deque([point])
            self.labels_[point] = cluster_id
            visited[point] = True
            
            while queue:
                current = queue.popleft()
                for neighbor in neighbors[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        self.labels_[neighbor] = cluster_id
                        if len(neighbors[neighbor]) >= self.min_samples:
                            queue.append(neighbor)
            
            cluster_id += 1
        
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
        return self

    def fit_predict(self, X: _Array) -> _Array:
        """Perform clustering on X and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point in the dataset. Noisy samples are given the label -1.
        """
        self.fit(X)
        return self.labels_

    def plot_clusters(self, X: _Array) -> None:
        """Plot DBSCAN cluster assignments for 2D data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Input samples. Must match fitted sample count.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != 2:
            raise ValueError("X must be two-dimensional with exactly 2 features.")
        if self.labels_.shape[0] == 0:
            raise ValueError("DBSCAN instance is not fitted yet. Call 'fit' first.")
        if X_arr.shape[0] != self.labels_.shape[0]:
            raise ValueError("X must have the same number of samples used during fitting.")

        # Import lazily so clustering can be used without plotting dependency.
        import matplotlib.pyplot as plt

        unique_labels = np.unique(self.labels_)
        cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

        plt.figure(figsize=(8, 6))
        for idx, label in enumerate(unique_labels):
            mask = self.labels_ == label
            if label == -1:
                plt.scatter(
                    X_arr[mask, 0],
                    X_arr[mask, 1],
                    c="k",
                    s=60,
                    marker="x",
                    label="Noise",
                )
            else:
                plt.scatter(
                    X_arr[mask, 0],
                    X_arr[mask, 1],
                    c=[cmap(idx)],
                    s=60,
                    label=f"Cluster {label}",
                )

        plt.title("DBSCAN Clustering Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
        