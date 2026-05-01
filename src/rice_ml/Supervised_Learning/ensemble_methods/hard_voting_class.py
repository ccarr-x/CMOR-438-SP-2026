from __future__ import annotations

from typing import Any, List, Optional, Union, Sequence, Tuple, Literal

from numpy.typing import NDArray
import numpy as np
from ..decision_tree_class import DecisionTreeClassifier
from ..decision_tree_regressor import DecisionTreeRegressor

import warnings

_Array = NDArray[Any]

class HardVotingClassifier:
    """
    A hard voting classifier that combines predictions from multiple classifiers.

    This classifier fits multiple base estimators and combines their predictions
    using majority voting. Similar to sklearn's VotingClassifier.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Base estimators to be used in the ensemble. Each estimator should be
        an unfitted classifier instance.

    Attributes
    ----------
    estimators_ : list of (str, estimator) tuples
        The fitted base estimators.
    """
    def __init__(self, estimators: List[Tuple[str, Any]]):
        self.estimators = estimators
        self.estimators_ = None

    def fit(self, X: _Array, y: _Array) -> 'HardVotingClassifier':
        """
        Fit the base estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : HardVotingClassifier
            Fitted classifier.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.estimators_ = []
        for name, estimator in self.estimators:
            estimator.fit(X, y)
            self.estimators_.append((name, estimator))
        return self

    def predict(self, X: _Array) -> _Array:
        """
        Predict class labels for X using majority voting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted class labels.
        """
        if self.estimators_ is None:
            raise ValueError("This HardVotingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        X = np.asarray(X)
        # Collect predictions from each fitted estimator
        predictions = np.array([estimator.predict(X) for _, estimator in self.estimators_])

        # Perform majority voting using base class method
        return self.aggregate_hard_voting(predictions)