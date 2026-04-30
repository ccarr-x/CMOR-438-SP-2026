"""Binary logistic regression as a thin wrapper over SingleNeuronModel."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from numpy.typing import NDArray

from .single_neuron_model_class import SingleNeuronModel

_Array = NDArray[Any]


class LogisticRegression:
    """Binary logistic regression with full-batch gradient descent.

    Parameters
    ----------
    eta : float, default=0.1
        Learning rate passed to :class:`GradientDescent`.
    epochs : int, default=200
        Number of gradient steps.
    random_state : int, optional
        Seed for reproducible initialization.

    Attributes
    ----------
    w_ : ndarray
        Learned feature weights of shape ``(n_features,)``.
    b_ : float
        Learned bias term.
    loss_history_ : list of float
        Binary cross-entropy recorded at each epoch (including initialization).
    param_history_ : list of ndarray
        Parameter vectors ``[w..., b]`` over optimization steps.
    """

    def __init__(
        self,
        eta: float = 0.1,
        epochs: int = 200,
        random_state: Optional[int] = None,
    ) -> None:
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.loss_history_: List[float] = []
        self.param_history_: List[_Array] = []
        self._model = SingleNeuronModel(
            task="logistic",
            eta=eta,
            epochs=epochs,
            random_state=random_state,
        )

    def train(self, X: _Array, y: _Array) -> LogisticRegression:
        """Fit logistic regression parameters by minimizing binary cross-entropy."""
        self._model.train(X, y)
        self.w_ = self._model.w_.copy()
        self.b_ = float(self._model.b_)
        self.loss_history_ = list(self._model.loss_history_)
        self.param_history_ = [p.copy() for p in self._model.param_history_]
        return self

    def net_input(self, X: _Array) -> Union[float, _Array]:
        """Compute linear scores ``X @ w + b``."""
        return self._model.net_input(X)

    def predict_proba(self, X: _Array) -> Union[float, _Array]:
        """Return estimated probability :math:`P(y=1|x)`."""
        return self._model.predict_proba(X)

    def predict(self, X: _Array) -> Union[int, _Array]:
        """Predict class label(s) using threshold 0.5 on probability."""
        return self._model.predict(X)
