"""Single-neuron linear regression as a thin wrapper over SingleNeuronModel."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from numpy.typing import NDArray

from .single_neuron_model_class import SingleNeuronModel

_Array = NDArray[Any]


class SingleNeuronLinearRegression:
    """Linear regression with one output neuron and additive bias.

    Parameters
    ----------
    eta : float, default=0.01
        Learning rate used by :class:`GradientDescent`.
    epochs : int, default=500
        Number of gradient steps.
    random_state : int, optional
        Seed for reproducible parameter initialization.

    Attributes
    ----------
    w_ : ndarray
        Learned feature weights with shape ``(n_features,)``.
    b_ : float
        Learned bias term.
    loss_history_ : list of float
        Mean squared error at each optimization iterate (includes iteration 0).
    param_history_ : list of ndarray
        Parameter vectors ``[w..., b]`` over optimization steps.
    """

    def __init__(
        self,
        eta: float = 0.01,
        epochs: int = 500,
        random_state: Optional[int] = None,
    ) -> None:
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.loss_history_: List[float] = []
        self.param_history_: List[_Array] = []
        self._model = SingleNeuronModel(
            task="linear",
            eta=eta,
            epochs=epochs,
            random_state=random_state,
        )

    def train(self, X: _Array, y: _Array) -> SingleNeuronLinearRegression:
        """Fit the linear neuron by minimizing mean squared error."""
        self._model.train(X, y)
        self.w_ = self._model.w_.copy()
        self.b_ = float(self._model.b_)
        self.loss_history_ = list(self._model.loss_history_)
        self.param_history_ = [p.copy() for p in self._model.param_history_]
        return self

    def net_input(self, X: _Array) -> Union[float, _Array]:
        """Compute linear output(s) ``X @ w + b``."""
        return self._model.net_input(X)

    def predict(self, X: _Array) -> Union[float, _Array]:
        """Predict target value(s) with the trained linear neuron."""
        return self.net_input(X)
