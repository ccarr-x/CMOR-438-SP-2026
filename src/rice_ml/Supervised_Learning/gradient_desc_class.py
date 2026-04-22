"""Numerical optimizers that step along a user-supplied gradient.

This module provides a small helper for iterative first-order optimization in
one or two variables. The one- and two-variable routines use *different update
signs* (see method docstrings): match them to whether ``gradient`` implements
an ascent or descent direction for your objective.
"""

from __future__ import annotations

from typing import Any, Callable, List, Union

import numpy as np
from numpy.typing import NDArray

_Array = NDArray[Any]
ObjectiveFn = Callable[..., Any]
GradientFn = Callable[[Any], Any]
_HistoryEntry = Union[float, _Array]


class GradientDescent:
    """Iterative updates from a callable gradient (one- or two-variable helpers).

    The objective ``function`` is stored for convenience but is not used by
    the optimization routines below; only ``gradient`` is evaluated each step.

    Parameters
    ----------
    function : callable
        Objective :math:`f` (unused by :meth:`optimize_1var` / :meth:`optimize_2var`).
    gradient : callable
        Returns the gradient :math:`\\nabla f` (or derivative for one dimension)
        at the current iterate.
    learning_rate : float
        Step size multiplying the gradient each iteration.
    num_iterations : int
        Number of gradient steps (plus the initial point, history length is
        ``num_iterations + 1``).

    Attributes
    ----------
    history : list
        Sequence of iterates after each call to an ``optimize_*`` method.
        Cleared and repopulated by that method.
    """

    def __init__(
        self,
        function: ObjectiveFn,
        gradient: GradientFn,
        learning_rate: float,
        num_iterations: int,
    ) -> None:
        self.function = function
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.history: List[_HistoryEntry] = []

    def optimize_1var(self, current_cost: float) -> None:
        """One-dimensional update: add ``learning_rate * gradient``.

        Starting from ``current_cost``, repeatedly applies::

            x <- x + learning_rate * gradient(x)

        Use this when ``gradient`` returns a direction consistent with *ascent*
        (or adjust the sign of ``learning_rate`` / ``gradient`` to minimize).

        Parameters
        ----------
        current_cost : float
            Initial scalar iterate :math:`x_0`.

        Notes
        -----
        Updates :attr:`history` in place and returns ``None``. The final iterate
        is ``self.history[-1]``.
        """
        self.history = [current_cost]

        for _ in range(self.num_iterations):
            gradient = self.gradient(current_cost)
            current_cost = current_cost + self.learning_rate * gradient
            self.history.append(current_cost)

    def optimize_2var(self, current_cost: _Array) -> _Array:
        """Two-variable (vector) update: subtract ``learning_rate * gradient``.

        Starting from ``current_cost``, repeatedly applies::

            x <- x - learning_rate * gradient(x)

        This matches standard gradient *descent* when ``gradient`` is
        :math:`\\nabla f`.

        Parameters
        ----------
        current_cost : ndarray
            Initial point, typically shape ``(2,)`` for two coordinates.

        Returns
        -------
        ndarray
            Array of shape ``(num_iterations + 1, ...)`` stacking every iterate
            (including the start) along axis 0.
        """
        self.history = [current_cost]

        for _ in range(self.num_iterations):
            gradient = self.gradient(current_cost)
            current_cost = current_cost - self.learning_rate * gradient
            self.history.append(current_cost)

        return np.array(self.history)
