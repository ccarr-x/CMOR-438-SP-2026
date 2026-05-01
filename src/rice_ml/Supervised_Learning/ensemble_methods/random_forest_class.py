from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np

from ..decision_tree_class import DecisionTreeClassifier
from ..decision_tree_regressor import DecisionTreeRegressor

import warnings

_Array = NDArray[Any]

class RandomForestClassifier:
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
        A random forest classifier.

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

    def fit(self, X: _Array, y: _Array) -> None:
        """
        Build a random forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create and fit a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
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
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        """
        X = np.asarray(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        return y_pred 

class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: Literal["mse", "mae"] = "mse",
        random_state: Optional[int] = None,
    ):
        """
        A random forest regressor.

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
        criterion : {"mse", "mae"}, default="mse"
            The function to measure the quality of a split. Supported criteria are
            "mse" for mean squared error, and "mae" for mean absolute error.
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
        self.trees: List[DecisionTreeRegressor] = []

    def fit(self, X: _Array, y: _Array) -> None:
        """
        Build a random forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create and fit a decision tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: _Array) -> _Array:
        """
        Predict regression target for X.

        The predicted regression target of an input sample is the average of the predictions of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted regression targets.
        """
        X = np.asarray(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Average predictions
        y_pred = np.mean(tree_preds, axis=0)
        return y_pred
    