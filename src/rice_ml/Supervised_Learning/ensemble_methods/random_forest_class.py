from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np

from ..decision_tree_class import DecisionTree
from ..decision_tree_regressor import DecisionTreeRegressor


_Array = NDArray[Any]

class _BaseEnsemble:
    """Base class for ensemble methods with common bootstrap logic."""
    
    def _bootstrap_fit(self, X: _Array, y: _Array, tree_class):
        """
        Common bootstrap fitting logic for ensemble methods.
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        tree_class : class
            The tree class to use (DecisionTree or DecisionTreeRegressor)
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
            tree = tree_class(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)


class RandomForestClassifier(_BaseEnsemble):
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
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node.
        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split.
        random_state : int, default=None
            Controls the randomness of the estimator.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.trees: List[DecisionTree] = []

    def fit(self, X: _Array, y: _Array) -> None:
        """Build a random forest classifier from the training set (X, y)."""
        self._bootstrap_fit(X, y, DecisionTree)

    def predict(self, X: _Array) -> _Array: 
        """
        Predict class for X.

        The predicted class of an input sample is a majority vote among the trees.
        """
        X = np.asarray(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        return y_pred 

class RandomForestRegressor(_BaseEnsemble):
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
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node.
        criterion : {"mse", "mae"}, default="mse"
            The function to measure the quality of a split.
        random_state : int, default=None
            Controls the randomness of the estimator.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.trees: List[DecisionTreeRegressor] = []

    def fit(self, X: _Array, y: _Array) -> None:
        """Build a random forest regressor from the training set (X, y)."""
        self._bootstrap_fit(X, y, DecisionTreeRegressor)

    def predict(self, X: _Array) -> _Array:
        """
        Predict regression target for X.

        The predicted value is the average of predictions from all trees.
        """
        X = np.asarray(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Average predictions
        y_pred = np.mean(tree_preds, axis=0)
        return y_pred
    