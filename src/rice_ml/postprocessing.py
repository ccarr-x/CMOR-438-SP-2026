"""Postprocessing utilities built on scikit-learn metrics."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
)

AverageType = Optional[Literal["binary", "micro", "macro", "weighted"]]
__all__ = ["PostProcessor"]


class PostProcessor:
    """Evaluate model predictions for classification and regression tasks."""

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        average: AverageType = "binary",
        labels: Optional[np.ndarray] = None,
        zero_division: int = 0,
    ) -> Dict[str, Any]:
        """Return classification error rate, precision, and confusion matrix."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")

        error_rate = float(np.mean(y_true_arr != y_pred_arr))
        precision = float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                average=average,
                zero_division=zero_division,
            )
        )
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

        return {
            "classification_error": error_rate,
            "precision": precision,
            "confusion_matrix": cm,
        }

    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        n_features: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return regression error metrics and ANOVA summary statistics."""
        y_true_arr = np.asarray(y_true, dtype=float).ravel()
        y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        if y_true_arr.size < 2:
            raise ValueError("At least two samples are required for regression metrics.")

        n = y_true_arr.size
        p = 1 if n_features is None else int(n_features)
        if p < 1:
            raise ValueError("n_features must be >= 1.")
        if n - p - 1 <= 0:
            raise ValueError("Need n_samples > n_features + 1 for ANOVA statistics.")

        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        mse = float(mean_squared_error(y_true_arr, y_pred_arr))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true_arr, y_pred_arr))

        y_mean = float(np.mean(y_true_arr))
        sst = float(np.sum((y_true_arr - y_mean) ** 2))
        sse = float(np.sum((y_true_arr - y_pred_arr) ** 2))
        ssr = max(sst - sse, 0.0)

        df_reg = float(p)
        df_err = float(n - p - 1)
        df_tot = float(n - 1)

        msr = ssr / df_reg
        mse_anova = sse / df_err
        f_stat = float("inf") if mse_anova == 0.0 else msr / mse_anova

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "sst": sst,
            "ssr": ssr,
            "sse": sse,
            "df_regression": df_reg,
            "df_error": df_err,
            "df_total": df_tot,
            "ms_regression": msr,
            "ms_error": mse_anova,
            "f_statistic": f_stat,
        }
