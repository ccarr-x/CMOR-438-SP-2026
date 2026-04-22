"""Binary perceptron classifier with online (stochastic) weight updates.

This module implements a single-layer perceptron for two-class problems with
labels in ``{-1, +1}``. Weights are updated per misclassified example using a
fixed learning rate until a maximum number of passes over the data or until
the training set is separated (zero errors in an epoch).
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
from numpy.typing import NDArray

# Alias for brevity in signatures
_Array = NDArray[Any]


class Perceptron(object):
    """Implementation of the perceptron algorithm with additive bias, 
    trained by stochastic weight updates. The perceptron algorithm is a 
    simple, supervised learning algorithm and linear classifier that aims 
    to linearly separate data into two classes.


    Parameters
    ----------
    eta : float, default=0.5
        Learning rate (step size) applied to each weight update when a
        sample is misclassified.
    epochs : int, default=20
        Maximum number of iterations over the training data. Training stops
        early if an epoch completes with no misclassifications.

    Attributes
    ----------
    w_ : ndarray
        Learned weights, size ``(n_features + 1,)``. The last element is the
        bias term
    errors_ : list of int
        Misclassification count per completed epoch (length <= ``epochs``). """

    def __init__(self, eta: float = 0.5, epochs: int = 20) -> None:
        self.eta = eta
        self.epochs = epochs

    def train(self, X: _Array, y: _Array) -> Perceptron:
        """Fit the perceptron to training data.

        Weights (including bias) are initialized at random. For each epoch,
        samples are visited in order, and each misclassified point triggers an
        update proportional to ``eta`` and the signed error.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels, expected to be ``-1`` or ``1`` for each row of ``X``.

        Returns
        -------
        self
            The fitted instance (for method chaining).
        """
        self.w_ = np.random.rand(1 + X.size[1])
        self.errors_: List[int] = []

        for _ in range(self.epochs):
            errors = 0
            for xi, label in zip(X, y):
                # Update rule: w <- w + eta * (y_true - y_pred) * x (bias)
                update = self.eta * (label - self.predict(xi))
                self.w_[:-1] += update * xi
                self.w_[-1] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def net_input(self, X: _Array) -> Union[float, _Array]:
        """Compute the linear score(s) before thresholding.

        For weights ``w`` (features) and bias ``b`` (last element of ``self.w_``),
        returns ``X @ w + b``. A single sample ``x`` yields a scalar; a batch
        ``X`` with shape ``(n_samples, n_features)`` yields a 1-D array of length
        ``n_samples``.

        Parameters
        ----------
        X : ndarray of shape (n_features,) or (n_samples, n_features)
            Feature vector(s).

        Returns
        -------
        float or ndarray
            Pre-activation value(s).
        """
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X: _Array) -> Union[int, _Array]:
        """Predict class label(s): ``1`` if net input >= 0, else ``-1``.

        Parameters
        ----------
        X : ndarray of shape (n_features,) or (n_samples, n_features)
            Feature vector(s).

        Returns
        -------
        int or ndarray
            Predicted label(s), same shape as the leading dimensions of ``X``.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
