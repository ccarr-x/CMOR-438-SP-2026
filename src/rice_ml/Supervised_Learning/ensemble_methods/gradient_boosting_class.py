from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np


_Array = NDArray[Any]

class GradientBoostingClassifier:
    """
    A gradient boosting classifier that combines predictions from multiple weak learners.

    This classifier fits multiple base estimators sequentially, where each estimator
    tries to correct the errors of the previous ones. Similar to sklearn's GradientBoostingClassifier.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by learning_rate.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    random_state : int, optional
        Seed for reproducible initialization.

    Attributes
    ----------
    estimators_ : list of fitted base estimators
        The fitted base estimators.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_: List[Any] = []

    def fit(self, X: _Array, y: _Array) -> 'GradientBoostingClassifier':
        """
        Fit the gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GradientBoostingClassifier
            Fitted classifier.
        """
        # Implementation of the fit method goes here
        return self

    def predict(self, X: _Array) -> _Array:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        # Implementation of the predict method goes here
        return np.empty(X.shape[0], dtype=int)

class GradientBoostingRegressor:
    """
    A gradient boosting regressor that combines predictions from multiple weak learners.

    This regressor fits multiple base estimators sequentially, where each estimator
    tries to correct the errors of the previous ones. Similar to sklearn's GradientBoostingRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by learning_rate.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    random_state : int, optional
        Seed for reproducible initialization.

    Attributes
    ----------
    estimators_ : list of fitted base estimators
        The fitted base estimators.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_: List[Any] = []

    def fit(self, X: _Array, y: _Array) -> 'GradientBoostingRegressor':
        """
        Fit the gradient boosting regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GradientBoostingRegressor
            Fitted regressor.
        """
        # Implementation of the fit method goes here
        return self

    def predict(self, X: _Array) -> _Array:
        """
        Predict regression values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted regression values.
        """
        # Implementation of the predict method goes here
        return np.empty(X.shape[0], dtype=float)
