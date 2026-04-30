"""Feedforward neural network with sigmoid activations and sigmoid cost.

This model supports binary classification with one or more hidden layers.
All layers use sigmoid activation, and training minimizes binary cross-entropy
("sigmoid cost") using full-batch gradient descent through GradientDescent.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from rice_ml.Supervised_Learning.gradient_desc_class import GradientDescent

_Array = NDArray[Any]


def _sigmoid(z: _Array) -> _Array:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class SigmoidNeuralNetwork:
    """Multi-layer feedforward network for binary classification.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, default=(8,)
        Number of units in each hidden layer.
    eta : float, default=0.1
        Learning rate passed to :class:`GradientDescent`.
    epochs : int, default=500
        Number of optimization steps.
    random_state : int, optional
        Seed for reproducible initialization.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (8,),
        eta: float = 0.1,
        epochs: int = 500,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.weights_: List[_Array] = []
        self.biases_: List[_Array] = []
        self.loss_history_: List[float] = []
        self.param_history_: List[_Array] = []
        self.layer_shapes_: Tuple[Tuple[int, int], ...] = ()

    def _prepare_targets(self, y: _Array) -> _Array:
        y = np.asarray(y, dtype=float).ravel()
        unique = set(np.unique(y))
        if unique <= {0.0, 1.0}:
            return y
        if unique <= {-1.0, 1.0}:
            return (y + 1.0) / 2.0
        raise ValueError("y must contain binary labels in {0, 1} or {-1, 1}.")

    def _init_parameters(self, n_features: int) -> Tuple[List[_Array], List[_Array]]:
        rng = np.random.default_rng(self.random_state)
        dims = (n_features,) + self.hidden_layer_sizes + (1,)
        weights: List[_Array] = []
        biases: List[_Array] = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros(fan_out, dtype=float)
            weights.append(w)
            biases.append(b)
        return weights, biases

    def _pack_params(self, weights: List[_Array], biases: List[_Array]) -> _Array:
        packed = [w.ravel() for w in weights] + [b.ravel() for b in biases]
        return np.concatenate(packed)

    def _unpack_params(self, params: _Array, dims: Tuple[int, ...]) -> Tuple[List[_Array], List[_Array]]:
        weights: List[_Array] = []
        biases: List[_Array] = []
        idx = 0
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            w_size = fan_in * fan_out
            w = params[idx : idx + w_size].reshape(fan_in, fan_out)
            idx += w_size
            weights.append(w)
        for i in range(len(dims) - 1):
            fan_out = dims[i + 1]
            b = params[idx : idx + fan_out]
            idx += fan_out
            biases.append(b)
        return weights, biases

    def _forward(self, X: _Array, weights: List[_Array], biases: List[_Array]) -> Tuple[List[_Array], List[_Array]]:
        activations: List[_Array] = [X]
        zs: List[_Array] = []
        a = X
        for w, b in zip(weights, biases):
            z = a @ w + b
            a = _sigmoid(z)
            zs.append(z)
            activations.append(a)
        return zs, activations

    def train(self, X: _Array, y: _Array) -> SigmoidNeuralNetwork:
        """Fit network parameters by full-batch gradient descent."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        y_bin = self._prepare_targets(y)
        if y_bin.shape[0] != X.shape[0]:
            raise ValueError("y must have one entry per row of X.")

        n_samples, n_features = X.shape
        dims = (n_features,) + self.hidden_layer_sizes + (1,)
        init_w, init_b = self._init_parameters(n_features)
        initial_params = self._pack_params(init_w, init_b)

        def objective(params: _Array) -> float:
            weights, biases = self._unpack_params(params, dims)
            _, activations = self._forward(X, weights, biases)
            probs = np.clip(activations[-1].ravel(), 1e-12, 1.0 - 1e-12)
            return -float(np.mean(y_bin * np.log(probs) + (1.0 - y_bin) * np.log(1.0 - probs)))

        def gradient(params: _Array) -> _Array:
            weights, biases = self._unpack_params(params, dims)
            _, activations = self._forward(X, weights, biases)

            y_col = y_bin.reshape(-1, 1)
            delta = (activations[-1] - y_col) / n_samples

            grad_w: List[_Array] = []
            grad_b: List[_Array] = []
            for layer_idx in range(len(weights) - 1, -1, -1):
                a_prev = activations[layer_idx]
                grad_w.insert(0, a_prev.T @ delta)
                grad_b.insert(0, np.sum(delta, axis=0))
                if layer_idx == 0:
                    break
                a_prev_hidden = activations[layer_idx]
                delta = (delta @ weights[layer_idx].T) * a_prev_hidden * (1.0 - a_prev_hidden)

            return self._pack_params(grad_w, grad_b)

        optimizer = GradientDescent(
            function=objective,
            gradient=gradient,
            learning_rate=self.eta,
            num_iterations=self.epochs,
        )
        history = optimizer.optimize_2var(initial_params)

        self.param_history_ = [p.copy() for p in history]
        self.loss_history_ = [objective(p) for p in history]
        final_w, final_b = self._unpack_params(history[-1], dims)
        self.weights_ = [w.copy() for w in final_w]
        self.biases_ = [b.copy() for b in final_b]
        self.layer_shapes_ = tuple((w.shape[0], w.shape[1]) for w in self.weights_)
        return self

    def predict_proba(self, X: _Array) -> Union[float, _Array]:
        """Return estimated probability :math:`P(y=1|x)`."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        _, activations = self._forward(X, self.weights_, self.biases_)
        probs = activations[-1].ravel()
        if single:
            return float(probs[0])
        return probs

    def predict(self, X: _Array) -> Union[int, _Array]:
        """Predict binary class label(s) with threshold 0.5."""
        probs = self.predict_proba(X)
        if isinstance(probs, float):
            return 1 if probs >= 0.5 else 0
        return np.where(probs >= 0.5, 1, 0)
