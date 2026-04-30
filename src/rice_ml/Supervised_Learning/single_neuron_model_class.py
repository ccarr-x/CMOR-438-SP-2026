"""Unified single-neuron model for linear or logistic objectives.

This class represents one neuron with affine pre-activation:

    z = X @ w + b

The neuron can be trained as either:
- linear regression (identity output, MSE loss), or
- logistic regression (sigmoid output, binary cross-entropy loss).

Optimization uses ``gradient_desc_class.GradientDescent``.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .gradient_desc_class import GradientDescent

_Array = NDArray[Any]
TaskType = Literal["linear", "logistic"]


def _sigmoid(z: _Array) -> _Array:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class SingleNeuronModel:
    """Single neuron with configurable learning objective.

    Parameters
    ----------
    task : {"linear", "logistic"}, default="linear"
        Training objective used for this neuron.
    eta : float, default=0.01
        Learning rate passed to :class:`GradientDescent`.
    epochs : int, default=500
        Number of gradient steps.
    random_state : int, optional
        Seed for reproducible initialization.
    """

    def __init__(
        self,
        task: TaskType = "linear",
        eta: float = 0.01,
        epochs: int = 500,
        random_state: Optional[int] = None,
    ) -> None:
        if task not in {"linear", "logistic"}:
            raise ValueError("task must be either 'linear' or 'logistic'.")
        self.task = task
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.loss_history_: List[float] = []
        self.param_history_: List[_Array] = []

    def _prepare_binary_targets(self, y: _Array) -> _Array:
        y = np.asarray(y, dtype=float).ravel()
        unique = set(np.unique(y))
        if unique <= {0.0, 1.0}:
            return y
        if unique <= {-1.0, 1.0}:
            return (y + 1.0) / 2.0
        raise ValueError("For logistic task, y must be in {0, 1} or {-1, 1}.")

    def train(self, X: _Array, y: _Array) -> SingleNeuronModel:
        """Fit the neuron with the configured objective."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")

        if self.task == "logistic":
            y_target = self._prepare_binary_targets(y)
        else:
            y_target = np.asarray(y, dtype=float).ravel()

        if y_target.shape[0] != X.shape[0]:
            raise ValueError("y must have one entry per row of X.")

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        initial_params = rng.normal(loc=0.0, scale=0.01, size=n_features + 1)

        def objective(params: _Array) -> float:
            z = X @ params[:-1] + params[-1]
            if self.task == "linear":
                return 0.5 * float(np.mean((z - y_target) ** 2))

            probs = _sigmoid(z)
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            return -float(
                np.mean(y_target * np.log(probs) + (1.0 - y_target) * np.log(1.0 - probs))
            )

        def gradient(params: _Array) -> _Array:
            z = X @ params[:-1] + params[-1]
            if self.task == "linear":
                residual = z - y_target
            else:
                residual = _sigmoid(z) - y_target
            grad_w = (X.T @ residual) / n_samples
            grad_b = np.array([np.mean(residual)])
            return np.concatenate((grad_w, grad_b))

        optimizer = GradientDescent(
            function=objective,
            gradient=gradient,
            learning_rate=self.eta,
            num_iterations=self.epochs,
        )
        history = optimizer.optimize_2var(initial_params)

        self.param_history_ = [p.copy() for p in history]
        self.loss_history_ = [objective(p) for p in history]
        final_params = history[-1]
        self.w_ = final_params[:-1].copy()
        self.b_ = float(final_params[-1])
        return self

    def net_input(self, X: _Array) -> Union[float, _Array]:
        """Return affine pre-activation output(s): ``X @ w + b``."""
        X = np.asarray(X, dtype=float)
        z = np.dot(X, self.w_) + self.b_
        if np.isscalar(z):
            return float(z)
        return z

    def predict_proba(self, X: _Array) -> Union[float, _Array]:
        """Return logistic probability; only valid when task='logistic'."""
        if self.task != "logistic":
            raise ValueError("predict_proba is only available when task='logistic'.")
        z = self.net_input(X)
        if isinstance(z, float):
            return float(_sigmoid(np.array(z)))
        return _sigmoid(z)

    def predict(self, X: _Array) -> Union[float, int, _Array]:
        """Predict output according to task type.

        - linear: returns continuous prediction(s)
        - logistic: returns class label(s) in {0, 1}
        """
        if self.task == "linear":
            return self.net_input(X)
        probs = self.predict_proba(X)
        if isinstance(probs, float):
            return 1 if probs >= 0.5 else 0
        return np.where(probs >= 0.5, 1, 0)
