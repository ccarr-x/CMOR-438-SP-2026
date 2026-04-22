"""Multi-layer perceptron (MLP) for binary classification.

This extends the single-layer :class:`perceptron_class.Perceptron` idea to a
stack of affine layers with nonlinear hidden units. Hidden layers use the
logistic sigmoid; the output layer is linear, and predictions use the same
``{-1, +1}`` threshold rule as the online perceptron. Weights are learned with
full-batch gradient descent on mean squared error between the scalar output
and the target label, using backpropagation through the hidden layers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[Any]


def _sigmoid(z: _Array) -> _Array:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_grad_from_a(a: _Array) -> _Array:
    """Derivative of sigmoid w.r.t. pre-activation, given activation ``a``."""
    return a * (1.0 - a)


class MultiLayerPerceptron:
    """Feedforward MLP with sigmoid hidden layers and a linear readout.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, default=(8,)
        Number of units in each hidden layer, ordered from input to output.
    eta : float, default=0.5
        Learning rate for batch gradient descent (same role as in
        :class:`perceptron_class.Perceptron`).
    epochs : int, default=200
        Number of full passes over the training set (MLPs typically need more
        passes than a linearly separable perceptron problem).
    random_state : int, optional
        Seed for weight initialization.

    Attributes
    ----------
    weights_ : list of ndarray
        Weight matrices ``W[l]`` with shape ``(fan_in, fan_out)`` for each
        layer, set by :meth:`train`.
    biases_ : list of ndarray
        Bias vectors ``b[l]``, each shape ``(fan_out,)``.
    layer_shapes_ : tuple of tuple of int
        ``((n_in, h0), (h0, h1), ..., (h_{k-1}, 1))`` describing the network
        after fitting.
    loss_history_ : list of float
        Mean squared error at the end of each epoch.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (8,),
        eta: float = 0.5,
        epochs: int = 200,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.weights_: List[_Array] = []
        self.biases_: List[_Array] = []
        self.layer_shapes_: Tuple[Tuple[int, int], ...] = ()
        self.loss_history_: List[float] = []

    def _init_parameters(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        dims = (n_features,) + self.hidden_layer_sizes + (1,)
        self.weights_ = []
        self.biases_ = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros(fan_out, dtype=float)
            self.weights_.append(w)
            self.biases_.append(b)
        self.layer_shapes_ = tuple((w.shape[0], w.shape[1]) for w in self.weights_)

    def _forward(self, X: _Array) -> Tuple[_Array, List[_Array], List[_Array]]:
        """Return output, list of pre-activations ``z`` (excluding input), activations ``a``."""
        activations: List[_Array] = [X]
        zs: List[_Array] = []
        a = X
        for layer_idx, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ w + b
            zs.append(z)
            if layer_idx < len(self.weights_) - 1:
                a = _sigmoid(z)
            else:
                a = z
            activations.append(a)
        out = activations[-1]
        return out, zs, activations

    def _backward(
        self,
        y: _Array,
        zs: List[_Array],
        activations: List[_Array],
    ) -> Tuple[List[_Array], List[_Array]]:
        m = float(y.shape[0])
        o = activations[-1]
        y_col = y.reshape(-1, 1)
        delta = (o - y_col) / m

        d_w: List[_Array] = []
        d_b: List[_Array] = []

        for layer_idx in range(len(self.weights_) - 1, -1, -1):
            a_prev = activations[layer_idx]
            d_w.insert(0, a_prev.T @ delta)
            d_b.insert(0, np.sum(delta, axis=0))

            if layer_idx == 0:
                break

            w = self.weights_[layer_idx]
            delta = (delta @ w.T) * _sigmoid_grad_from_a(activations[layer_idx])

        return d_w, d_b

    def train(self, X: _Array, y: _Array) -> MultiLayerPerceptron:
        """Fit the MLP by batch gradient descent with backpropagation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Targets in ``{-1, +1}`` (same convention as :class:`perceptron_class.Perceptron`).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have one entry per row of X.")

        self._init_parameters(X.shape[1])
        self.loss_history_ = []

        for _ in range(self.epochs):
            out, zs, activations = self._forward(X)
            loss = 0.5 * float(np.mean((out.ravel() - y) ** 2))
            self.loss_history_.append(loss)

            d_w, d_b = self._backward(y, zs, activations)
            for i in range(len(self.weights_)):
                self.weights_[i] -= self.eta * d_w[i]
                self.biases_[i] -= self.eta * d_b[i]

        return self

    def net_input(self, X: _Array) -> Union[float, _Array]:
        """Final linear output before thresholding (analogous to the perceptron score).

        For a single sample of shape ``(n_features,)``, returns a scalar. For a
        batch ``(n_samples, n_features)``, returns a 1-D array of length
        ``n_samples``.
        """
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        out, _, _ = self._forward(X)
        out_1d = out.ravel()
        if single:
            return float(out_1d[0])
        return out_1d

    def predict(self, X: _Array) -> Union[int, _Array]:
        """Predict ``1`` if net input >= 0, else ``-1`` (same rule as the perceptron)."""
        scores = self.net_input(X)
        if isinstance(scores, float):
            return 1 if scores >= 0.0 else -1
        return np.where(scores >= 0.0, 1, -1)
