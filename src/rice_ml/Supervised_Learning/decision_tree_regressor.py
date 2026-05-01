from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np

import warnings

_Array = NDArray[Any]

"""
Decision Tree Regressor Module

This module provides a from-scratch implementation of a decision tree regressor,
similar to scikit-learn's DecisionTreeRegressor. It supports regression tasks
using mean squared error (MSE) or mean absolute error (MAE) as splitting criteria.

The implementation includes features like:
- Configurable maximum depth and minimum samples for splitting/leaves
- Support for both MSE and MAE splitting criteria
- Efficient threshold selection for continuous features
- Input validation and error handling

Example
-------
>>> from decision_tree_regressor import DecisionTreeRegressor
>>> import numpy as np
>>> X = np.array([[1, 2], [3, 4], [5, 6]])
>>> y = np.array([1.0, 2.0, 3.0])
>>> regressor = DecisionTreeRegressor(max_depth=2)
>>> regressor.fit(X, y)
>>> predictions = regressor.predict(X)
"""

class DecisionTreeRegressor:
    """
    A decision tree regressor built from scratch.

    This implementation recursively builds a decision tree by selecting the best
    feature and threshold to split the data, minimizing the chosen criterion
    (MSE or MAE). The tree is represented as nested tuples.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    criterion : {"mse", "mae"}, default="mse"
        The function to measure the quality of a split. Supported criteria are
        "mse" for the mean squared error, and "mae" for the mean absolute error.

    Attributes
    ----------
    tree : tuple or float
        The fitted decision tree. Internal nodes are represented as tuples
        (feature_index, threshold, left_subtree, right_subtree), and leaves
        are represented as floats (the predicted value).

    Examples
    --------
    >>> import numpy as np
    >>> from decision_tree_regressor import DecisionTreeRegressor
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = np.array([0.0, 1.0])
    >>> regressor = DecisionTreeRegressor(max_depth=1)
    >>> regressor.fit(X, y)
    >>> regressor.predict([[0.5, 0.5]])
    array([0.5])
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="mse"):
        """
        Initialize the DecisionTreeRegressor.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples in a leaf node.
        criterion : str, default="mse"
            Splitting criterion: "mse" or "mae".
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None

    def fit(self, X: _Array, y: _Array) -> None:
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X: _Array, y: _Array, depth: int = 0) -> Union[float, Tuple]:
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : array-like
            Feature matrix for the current node.
        y : array-like
            Target values for the current node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        tree : float or tuple
            A leaf node (float) or internal node (tuple).
        """
        n_samples, n_features = X.shape

        # Check stopping criteria
        if (depth == self.max_depth or 
            n_samples < self.min_samples_split or 
            np.var(y) == 0):  # All targets are the same
            return np.mean(y)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            return np.mean(y)

        # Create left and right subsets
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Ensure minimum samples per leaf
        if (left_indices.sum() < self.min_samples_leaf or 
            right_indices.sum() < self.min_samples_leaf):
            return np.mean(y)

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X: _Array, y: _Array, n_features: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split the data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target values.
        n_features : int
            Number of features.

        Returns
        -------
        best_feature : int or None
            Index of the best feature to split on.
        best_threshold : float or None
            Best threshold value for the split.
        """
        best_score = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = self._get_thresholds(X[:, feature])
            for threshold in thresholds:
                score = self._calculate_score(X, y, feature, threshold)
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _get_thresholds(self, feature_values: _Array) -> _Array:
        """
        Get potential split thresholds for a feature by computing midpoints.

        Parameters
        ----------
        feature_values : array-like
            Values of a single feature.

        Returns
        -------
        thresholds : array
            Array of threshold values (midpoints between consecutive unique values).
        """
        sorted_values = np.unique(feature_values)
        if len(sorted_values) <= 1:
            return np.array([])
        return (sorted_values[:-1] + sorted_values[1:]) / 2

    def _calculate_score(self, X: _Array, y: _Array, feature: int, threshold: float) -> float:
        """
        Calculate the splitting score for a given feature and threshold.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target values.
        feature : int
            Feature index.
        threshold : float
            Split threshold.

        Returns
        -------
        score : float
            The calculated score (MSE or MAE weighted by sample sizes).
        """
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return float('inf')

        if self.criterion == "mse":
            left_mean = np.mean(y[left_indices])
            right_mean = np.mean(y[right_indices])
            mse_left = np.mean((y[left_indices] - left_mean) ** 2)
            mse_right = np.mean((y[right_indices] - right_mean) ** 2)
            return (left_indices.sum() * mse_left + right_indices.sum() * mse_right) / len(y)
        elif self.criterion == "mae":
            left_median = np.median(y[left_indices])
            right_median = np.median(y[right_indices])
            mae_left = np.mean(np.abs(y[left_indices] - left_median))
            mae_right = np.mean(np.abs(y[right_indices] - right_median))
            return (left_indices.sum() * mae_left + right_indices.sum() * mae_right) / len(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _predict_single(self, x: _Array, tree: Union[float, Tuple]) -> float:
        """
        Predict the target value for a single sample by traversing the tree.

        Parameters
        ----------
        x : array-like
            A single sample's features.
        tree : float or tuple
            The decision tree structure.

        Returns
        -------
        prediction : float
            The predicted target value.
        """
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_tree, right_tree = tree
        if x[feature] < threshold:
            return self._predict_single(x, left_tree)
        else:
            return self._predict_single(x, right_tree)
        
    def predict(self, X: _Array) -> _Array:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        X = np.asarray(X)
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def score(self, X: _Array, y: _Array) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return 1 - np.mean((y - y_pred) ** 2) / np.var(y)