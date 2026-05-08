"""Postprocessing utilities for classification and regression evaluation.

Classification precision, recall, and F1 are computed with NumPy from a confusion matrix.
Regression metrics and ROC helpers use :mod:`sklearn.metrics`.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve

AverageType = Optional[Literal["binary", "micro", "macro", "weighted"]]
__all__ = ["PostProcessor"]


class PostProcessor:
    """Evaluate model predictions for classification and regression tasks."""

    @staticmethod
    def _confusion_matrix_np(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build confusion matrix with NumPy only."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")

        if labels is None:
            labels_arr = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
        else:
            labels_arr = np.asarray(labels)
            if labels_arr.ndim != 1:
                raise ValueError("labels must be a 1D array if provided.")

        label_to_idx = {label: idx for idx, label in enumerate(labels_arr.tolist())}
        cm = np.zeros((labels_arr.size, labels_arr.size), dtype=int)

        for yt, yp in zip(y_true_arr, y_pred_arr):
            if yt not in label_to_idx or yp not in label_to_idx:
                continue
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

        return cm, labels_arr

    @staticmethod
    def _precision_recall_f1_from_cm(
        cm: np.ndarray,
        average: AverageType = "binary",
        zero_division: int = 0,
    ) -> tuple[float, float, float]:
        """Compute precision/recall/F1 from confusion matrix."""
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        support = cm.sum(axis=1).astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision_per_class = np.divide(
                tp,
                tp + fp,
                out=np.full_like(tp, float(zero_division)),
                where=(tp + fp) != 0,
            )
            recall_per_class = np.divide(
                tp,
                tp + fn,
                out=np.full_like(tp, float(zero_division)),
                where=(tp + fn) != 0,
            )
            f1_per_class = np.divide(
                2.0 * precision_per_class * recall_per_class,
                precision_per_class + recall_per_class,
                out=np.full_like(tp, float(zero_division)),
                where=(precision_per_class + recall_per_class) != 0,
            )

        if average == "binary":
            if cm.shape[0] != 2:
                raise ValueError("average='binary' requires exactly 2 classes.")
            return (
                float(precision_per_class[1]),
                float(recall_per_class[1]),
                float(f1_per_class[1]),
            )

        if average == "micro":
            tp_sum = float(tp.sum())
            fp_sum = float(fp.sum())
            fn_sum = float(fn.sum())
            precision_micro = (
                tp_sum / (tp_sum + fp_sum)
                if (tp_sum + fp_sum) != 0
                else float(zero_division)
            )
            recall_micro = (
                tp_sum / (tp_sum + fn_sum)
                if (tp_sum + fn_sum) != 0
                else float(zero_division)
            )
            f1_micro = (
                2.0 * precision_micro * recall_micro / (precision_micro + recall_micro)
                if (precision_micro + recall_micro) != 0
                else float(zero_division)
            )
            return float(precision_micro), float(recall_micro), float(f1_micro)

        if average == "macro":
            return (
                float(np.mean(precision_per_class)),
                float(np.mean(recall_per_class)),
                float(np.mean(f1_per_class)),
            )

        if average == "weighted":
            total_support = float(support.sum())
            if total_support == 0:
                return float(zero_division), float(zero_division), float(zero_division)
            return (
                float(np.sum(precision_per_class * support) / total_support),
                float(np.sum(recall_per_class * support) / total_support),
                float(np.sum(f1_per_class * support) / total_support),
            )

        raise ValueError("average must be one of: 'binary', 'micro', 'macro', or 'weighted'.")

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[np.ndarray] = None,
        *,
        average: AverageType = "binary",
        zero_division: int = 0,
    ) -> Dict[str, Any]:
        """Return classification error rate, precision, and confusion matrix."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")

        error_rate = float(np.mean(y_true_arr != y_pred_arr))
        cm, _ = self._confusion_matrix_np(y_true_arr, y_pred_arr, labels=labels)
        precision_value, recall_value, f1_value = self._precision_recall_f1_from_cm(
            cm, average=average, zero_division=zero_division
        )

        return {
            "classification_error": error_rate,
            "precision": precision_value,
            "recall": recall_value,
            "f1_score": f1_value,
            "confusion_matrix": cm,
        }

    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
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
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Return confusion matrix for classification."""
        cm, _ = self._confusion_matrix_np(y_true, y_pred, labels=labels)
        return cm
    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Plot confusion matrix for classification."""
        sns.set_theme()
        cm = self.confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        return cm
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return classification accuracy."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        return float(np.mean(y_true_arr == y_pred_arr))
    
    def precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: AverageType = "binary", zero_division: int = 0) -> float:
        """Return classification precision."""
        cm, _ = self._confusion_matrix_np(y_true, y_pred)
        precision_value, _, _ = self._precision_recall_f1_from_cm(
            cm, average=average, zero_division=zero_division
        )
        return precision_value

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray, average: AverageType = "binary", zero_division: int = 0) -> float:
        """Return classification recall."""
        cm, _ = self._confusion_matrix_np(y_true, y_pred)
        _, recall_value, _ = self._precision_recall_f1_from_cm(
            cm, average=average, zero_division=zero_division
        )
        return recall_value

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: AverageType = "binary", zero_division: int = 0) -> float:
        """Return classification F1 score."""
        cm, _ = self._confusion_matrix_np(y_true, y_pred)
        _, _, f1_value = self._precision_recall_f1_from_cm(
            cm, average=average, zero_division=zero_division
        )
        return f1_value

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return regression R² score."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        return float(r2_score(y_true_arr, y_pred_arr))

    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return mean absolute error."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        return float(mean_absolute_error(y_true_arr, y_pred_arr))
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return mean squared error."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        return float(mean_squared_error(y_true_arr, y_pred_arr))

    def roc_auc_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return ROC AUC score."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        return float(roc_auc_score(y_true_arr, y_pred_arr))

    def roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ROC curve."""
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        return roc_curve(y_true_arr, y_pred_arr)

    def plot_roc_curve(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plot ROC curve."""
        sns.set_theme()
        fpr, tpr, thresholds = self.roc_curve(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        return fpr, tpr, thresholds
