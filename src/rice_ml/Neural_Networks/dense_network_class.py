"""Dense feedforward network trained by SGD with backpropagation and MSE.

Hidden layers use sigmoid activation. The output layer is linear so the model
can learn continuous targets with mean squared error.
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


class DenseNetwork:
    """Dense network for regression with MSE and SGD.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, default=(8,)
        Number of units in each hidden layer.
    eta : float, default=0.01
        Learning rate for SGD updates.
    epochs : int, default=200
        Number of full passes over the training set.
    random_state : int, optional
        Seed for initialization and SGD shuffling.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (8,),
        eta: float = 0.01,
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
        self.param_history_: List[_Array] = []

    def _init_parameters(self, n_features: int) -> Tuple[List[_Array], List[_Array], Tuple[int, ...]]:
        rng = np.random.default_rng(self.random_state)
        dims = (n_features,) + self.hidden_layer_sizes + (1,)
        weights: List[_Array] = []
        biases: List[_Array] = []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            biases.append(np.zeros(fan_out, dtype=float))
        return weights, biases, dims

    def _pack_params(self, weights: List[_Array], biases: List[_Array]) -> _Array:
        chunks = [w.ravel() for w in weights] + [b.ravel() for b in biases]
        return np.concatenate(chunks)

    def _unpack_params(self, params: _Array, dims: Tuple[int, ...]) -> Tuple[List[_Array], List[_Array]]:
        weights: List[_Array] = []
        biases: List[_Array] = []
        idx = 0
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            n_w = fan_in * fan_out
            weights.append(params[idx : idx + n_w].reshape(fan_in, fan_out))
            idx += n_w
        for i in range(len(dims) - 1):
            fan_out = dims[i + 1]
            biases.append(params[idx : idx + fan_out])
            idx += fan_out
        return weights, biases

    def _forward(self, X: _Array, weights: List[_Array], biases: List[_Array]) -> Tuple[List[_Array], List[_Array]]:
        activations: List[_Array] = [X]
        zs: List[_Array] = []
        a = X
        for layer_idx, (w, b) in enumerate(zip(weights, biases)):
            z = a @ w + b
            zs.append(z)
            if layer_idx < len(weights) - 1:
                a = _sigmoid(z)
            else:
                a = z
            activations.append(a)
        return z, activations

    def train(self, X: _Array, y: _Array) -> DenseNetwork:
        """Fit network weights with SGD and sample-wise backpropagation."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional (n_samples, n_features).")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have one entry per row of X.")

        n_samples, n_features = X.shape
        init_w, init_b, dims = self._init_parameters(n_features)
        initial_params = self._pack_params(init_w, init_b)

        def objective(params: _Array) -> float:
            weights, biases = self._unpack_params(params, dims)
            _, activations = self._forward(X, weights, biases)
            preds = activations[-1].ravel()
            return 0.5 * float(np.mean((preds - y) ** 2))

        def sample_gradient(params: _Array, sample_idx: int) -> _Array:
            weights, biases = self._unpack_params(params, dims)
            xi = X[sample_idx : sample_idx + 1]
            yi = y[sample_idx]

            _, activations = self._forward(xi, weights, biases)
            pred = activations[-1]  # shape (1, 1)
            delta = pred - yi

            grad_w: List[_Array] = []
            grad_b: List[_Array] = []
            for layer_idx in range(len(weights) - 1, -1, -1):
                a_prev = activations[layer_idx]
                grad_w.insert(0, a_prev.T @ delta)
                grad_b.insert(0, np.sum(delta, axis=0))
                if layer_idx == 0:
                    break
                a_hidden = activations[layer_idx]
                delta = (delta @ weights[layer_idx].T) * a_hidden * (1.0 - a_hidden)

            return self._pack_params(grad_w, grad_b)

        optimizer = GradientDescent(
            function=objective,
            gradient=sample_gradient,
            learning_rate=self.eta,
            num_iterations=self.epochs,
        )
        history = optimizer.stochastic_gd(
            initial_params,
            n_samples=n_samples,
            random_state=self.random_state,
            shuffle=True,
        )

        self.param_history_ = [p.copy() for p in history]
        self.loss_history_ = [objective(p) for p in history]
        final_w, final_b = self._unpack_params(history[-1], dims)
        self.weights_ = [w.copy() for w in final_w]
        self.biases_ = [b.copy() for b in final_b]
        self.layer_shapes_ = tuple((w.shape[0], w.shape[1]) for w in self.weights_)
        return self

    def predict(self, X: _Array) -> Union[float, _Array]:
        """Predict continuous output(s)."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        _, activations = self._forward(X, self.weights_, self.biases_)
        preds = activations[-1].ravel()
        if single:
            return float(preds[0])
        return preds
