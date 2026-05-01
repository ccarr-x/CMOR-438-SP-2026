from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np

import warnings

_Array = NDArray[Any]

"""
Decision Tree Classifier Module

This module provides a from-scratch implementation of a decision tree classifier,
similar to scikit-learn's DecisionTreeClassifier. It supports classification tasks
using Gini impurity or entropy as splitting criteria.

The implementation includes features like:
- Configurable maximum depth and minimum samples for splitting/leaves
- Support for Gini impurity and entropy splitting criteria
- Efficient threshold selection for continuous features
- Input validation and error handling

Example
-------
>>> from decision_tree_class import DecisionTree
>>> import numpy as np
>>> X = np.array([[1, 2], [3, 4], [5, 6]])
>>> y = np.array([0, 1, 1])
>>> classifier = DecisionTree(max_depth=2)
>>> classifier.fit(X, y)
>>> predictions = classifier.predict(X)
"""

class DecisionTree:
    """
    A decision tree classifier built from scratch.

    This implementation recursively builds a decision tree by selecting the best
    feature and threshold to split the data, minimizing the chosen impurity criterion
    (Gini or entropy). The tree is represented as nested tuples.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity, and "entropy" for the information gain.

    Attributes
    ----------
    tree : tuple or int
        The fitted decision tree. Internal nodes are represented as tuples
        (feature_index, threshold, left_subtree, right_subtree), and leaves
        are represented as integers (the predicted class).

    Examples
    --------
    >>> import numpy as np
    >>> from decision_tree_class import DecisionTree
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = np.array([0, 1])
    >>> classifier = DecisionTree(max_depth=1)
    >>> classifier.fit(X, y)
    >>> classifier.predict([[0.5, 0.5]])
    array([0])
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        """
        Initialize the DecisionTree classifier.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples in a leaf node.
        criterion : str, default="gini"
            Splitting criterion: "gini" or "entropy".
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
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
        tree : int or tuple
            A leaf node (int) or internal node (tuple).
        """
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Check stopping criteria
        if (len(unique_classes) == 1 or 
            depth == self.max_depth or 
            n_samples < self.min_samples_split):
            return unique_classes[0]

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            return unique_classes[0]

        # Create left and right subsets
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Ensure minimum samples per leaf
        if (left_indices.sum() < self.min_samples_leaf or 
            right_indices.sum() < self.min_samples_leaf):
            return unique_classes[0]

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X, y, n_features):
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
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = self._get_thresholds(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _get_thresholds(self, feature_values):
        """Get potential split thresholds for a feature."""
        sorted_values = np.unique(feature_values)
        if len(sorted_values) <= 1:
            return np.array([])
        return (sorted_values[:-1] + sorted_values[1:]) / 2

    def _information_gain(self, X, y, feature, threshold):
        """
        Calculate the information gain for a given feature and threshold.

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
        gain : float
            The information gain.
        """
        # Calculate the information gain based on the chosen criterion
        if self.criterion == 'gini':
            return self._gini_gain(X, y, feature, threshold)
        elif self.criterion == 'entropy':
            return self._entropy_gain(X, y, feature, threshold)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _gini_gain(self, X, y, feature, threshold):
        """
        Calculate the Gini impurity gain for a split.

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
        gain : float
            The Gini gain.
        """
        # Calculate Gini impurity and gain
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        left_size = left_indices.sum()
        right_size = right_indices.sum()

        if left_size == 0 or right_size == 0:
            return 0

        p_left = left_size / len(y)
        p_right = right_size / len(y)

        gini_left = 1 - sum((np.bincount(y[left_indices]) / left_size) ** 2)
        gini_right = 1 - sum((np.bincount(y[right_indices]) / right_size) ** 2)

        gini_gain = (p_left * gini_left + p_right * gini_right)
        return gini_gain

    def _entropy_gain(self, X, y, feature, threshold):
        """
        Calculate the entropy gain for a split.

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
        gain : float
            The entropy gain.
        """
        # Calculate entropy and gain
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        left_size = left_indices.sum()
        right_size = right_indices.sum()

        if left_size == 0 or right_size == 0:
            return 0

        p_left = left_size / len(y)
        p_right = right_size / len(y)

        def entropy(labels):
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / len(labels)
            return -sum(probs * np.log2(probs + 1e-10))  # Add small value to avoid log(0)

        entropy_left = entropy(y[left_indices])
        entropy_right = entropy(y[right_indices])

        entropy_gain = entropy(y) - (p_left * entropy_left + p_right * entropy_right)
        return entropy_gain

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted class labels.
        """
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _predict_sample(self, sample, tree):
        """
        Predict the class label for a single sample by traversing the tree.

        Parameters
        ----------
        sample : array-like
            A single sample's features.
        tree : int or tuple
            The decision tree structure.

        Returns
        -------
        prediction : int
            The predicted class label.
        """
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_tree, right_tree = tree
        if sample[feature] < threshold:
            return self._predict_sample(sample, left_tree)
        else:
            return self._predict_sample(sample, right_tree)
