from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np

from ..decision_tree_class import DecisionTreeClassifier
from ..decision_tree_regressor import DecisionTreeRegressor

import warnings

_Array = NDArray[Any]

class BaggingForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: Literal["gini", "entropy"] = "gini",
        random_state: Optional[int] = None,
    ):
        """
        A bagging forest classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
        max_depth : int, default=None
            The maximum depth of the tree. 
            If None, then nodes are expanded until all leaves are pure 
            or until all leaves contain less than min_samples_split samples.
            min_samples_split : int, default=2
                The minimum number of samples required to split an internal node.
            min_samples_leaf : int, default=1
                The minimum number of samples required to be at a leaf node.
        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity, and "entropy" for the information gain.
        random_state : int, default=None
            Controls the randomness of the estimator. The features are always randomly permuted at each split.
            When max_features < n_features, the algorithm will select max_features at random for each split before finding the best split among them.
            So, random_state affects the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifier] = []

    def aggregate(self, predictions: _Array) -> _Array:
        """
        Aggregate predictions from individual trees using majority voting.

        Parameters
        ----------
        predictions : array-like of shape (n_estimators, n_samples)
            The predictions from each tree for each sample.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The aggregated predicted class labels.
        """
        # predictions shape: (n_estimators, n_samples)
        n_samples = predictions.shape[1]
        y_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = predictions[:, i]
            y_pred[i] = np.bincount(votes).argmax()
        return y_pred
    
    def bootstrap(self, X: _Array, y: _Array) -> Tuple[_Array, _Array]:
        """
        Generate a bootstrap sample from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        X_bootstrap : array-like of shape (n_samples, n_features)
            The bootstrap sample of the input samples.
        y_bootstrap : array-like of shape (n_samples,)
            The bootstrap sample of the target values.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap

    def fit(self, X: _Array, y: _Array) -> None:
        """
        Build a bagging forest of trees from the training set (X, y).

        Parameters
        ----------
         X : array-like of shape (n_samples, n_features)
             The training input samples. Internally, 
             it will be converted to an array if it's 
             not already an array.
        y : array-like of shape (n_samples,)
             The target values (class labels). Internally, 
             it will be converted to an array if it's 
             not already an array.
             """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if self.random_state is not None:
            np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_bootstrap, y_bootstrap = self.bootstrap(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                random_state=None,  # Each tree should have its own random state
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    def predict(self, X: _Array) -> _Array:
        """
        Predict class for X.

        The predicted class of an input sample is a majority vote among the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to an array if it's not already an array.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        # Collect predictions from all trees
        predictions = np.zeros((self.n_estimators, n_samples), dtype=int)
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)
        # Aggregate using majority vote
        return self.aggregate(predictions)

    def score(self, X: _Array, y: _Array) -> float:
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
    