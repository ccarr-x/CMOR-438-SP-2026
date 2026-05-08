#!/usr/bin/env python3
"""Train multiple ``rice_ml`` classifiers on the spam email dataset and compare test metrics.

Uses ``examples/load_dataset.py`` for portable paths. Feature columns match the
spam-focused supervised notebooks (numeric signals only). All models see the
same train/test split and features standardized with training-set statistics.

Run from the repository root (recommended)::

    python examples/compare_spam_models.py

Or from ``examples/``::

    python compare_spam_models.py

The same evaluation is available as a notebook: ``examples/compare_spam_models.ipynb``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ``src/`` on path so ``rice_ml`` imports when run as ``python examples/compare_spam_models.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SPAM_CSV = Path(__file__).resolve().parent / "datasets" / "spam_email_dataset.csv"


def load_spam_dataset():
    """Load spam CSV next to this script (same path as :func:`examples.load_dataset.load_spam_dataset`)."""
    if not _SPAM_CSV.is_file():
        raise FileNotFoundError(f"Spam dataset not found at {_SPAM_CSV}")
    return pd.read_csv(_SPAM_CSV)
from rice_ml.Neural_Networks.neural_network_class import SigmoidNeuralNetwork  # noqa: E402
from rice_ml.Supervised_Learning.decision_tree_class import DecisionTree  # noqa: E402
from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import (  # noqa: E402
    RandomForestClassifier,
)
from rice_ml.Supervised_Learning.knn_class import KNearestNeighbors  # noqa: E402
from rice_ml.Supervised_Learning.logistic_regression_class import LogisticRegression  # noqa: E402
from rice_ml.Supervised_Learning.multilayer_perceptron_class import MultiLayerPerceptron  # noqa: E402
from rice_ml.Supervised_Learning.perceptron_class import Perceptron  # noqa: E402
from rice_ml.processing.postprocessing import PostProcessor  # noqa: E402
from rice_ml.processing.preprocessing import standardize, train_test_split  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None  # type: ignore[misc, assignment]

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    tabulate = None  # type: ignore[misc, assignment]

# Same numeric features used across Decision_Trees / KNN / MLP spam examples.
FEATURES = [
    "num_words",
    "num_links",
    "has_suspicious_link",
    "num_attachments",
    "sender_reputation_score",
]
TARGET = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _to_binary_labels_pm1(y: np.ndarray) -> np.ndarray:
    """Map {0, 1} -> {-1, +1} for :class:`Perceptron`."""
    y_arr = np.asarray(y).ravel()
    return np.where(y_arr == 0, -1, 1)


def _from_pm1(y: np.ndarray) -> np.ndarray:
    """Map {-1, +1} -> {0, 1} for evaluation."""
    y_arr = np.asarray(y).ravel()
    return np.where(y_arr < 0, 0, 1).astype(int)


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    if roc_auc_score is None:
        return None
    y_arr = np.asarray(y_true).ravel()
    p = np.asarray(proba).ravel()
    if len(np.unique(y_arr)) < 2:
        return None
    return float(roc_auc_score(y_arr, p))


def build_rows(
    quick: bool,
) -> List[Tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], Optional[Callable[..., Any]]]]:
    """Return (model_name, predict_fn, optional_proba_fn) factories."""

    def pred_logistic(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = LogisticRegression(eta=0.15, epochs=500 if not quick else 200, random_state=RANDOM_STATE)
        m.train(X_tr, y_tr)
        return np.asarray(m.predict(X_te), dtype=int)

    def proba_logistic(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = LogisticRegression(eta=0.15, epochs=500 if not quick else 200, random_state=RANDOM_STATE)
        m.train(X_tr, y_tr)
        return np.asarray(m.predict_proba(X_te), dtype=float).ravel()

    def pred_perceptron(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        y_pm = _to_binary_labels_pm1(y_tr)
        m = Perceptron(eta=0.01, epochs=300 if not quick else 80)
        m.train(X_tr, y_pm)
        return _from_pm1(m.predict(X_te))

    def pred_tree(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = DecisionTree(max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion="gini")
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        return np.asarray(pred, dtype=int).ravel()

    def pred_knn(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = KNearestNeighbors(n_neighbors=5, task="classification")
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_te), dtype=int).ravel()

    def pred_rf(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        n_est = 30 if quick else 100
        m = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=5,
            random_state=RANDOM_STATE,
        )
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_te), dtype=int).ravel()

    def pred_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = MultiLayerPerceptron(
            hidden_layer_sizes=(16,) if not quick else (8,),
            eta=0.1,
            epochs=200 if not quick else 80,
            random_state=RANDOM_STATE,
        )
        m.train(X_tr, y_tr)
        return np.asarray(m.predict(X_te), dtype=int).ravel()

    def pred_snn(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = SigmoidNeuralNetwork(
            hidden_layer_sizes=(16,) if not quick else (8,),
            eta=0.1,
            epochs=300 if not quick else 100,
            random_state=RANDOM_STATE,
        )
        m.train(X_tr, y_tr)
        return np.asarray(m.predict(X_te), dtype=int).ravel()

    def proba_snn(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        m = SigmoidNeuralNetwork(
            hidden_layer_sizes=(16,) if not quick else (8,),
            eta=0.1,
            epochs=300 if not quick else 100,
            random_state=RANDOM_STATE,
        )
        m.train(X_tr, y_tr)
        return np.asarray(m.predict_proba(X_te), dtype=float).ravel()

    rows: List[
        Tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], Optional[Callable[..., Any]]]
    ] = [
        ("LogisticRegression", pred_logistic, proba_logistic),
        ("Perceptron", pred_perceptron, None),
        ("DecisionTree", pred_tree, None),
        ("KNearestNeighbors (k=5)", pred_knn, None),
        ("RandomForestClassifier", pred_rf, None),
        ("MultiLayerPerceptron", pred_mlp, None),
        ("SigmoidNeuralNetwork", pred_snn, proba_snn),
    ]
    return rows


def run_comparison(quick: bool = False) -> pd.DataFrame:
    """Load spam data, train each model, return a DataFrame of test-set metrics."""
    df_data = load_spam_dataset()
    missing = [c for c in FEATURES + [TARGET] if c not in df_data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df_data[FEATURES].values.astype(float)
    y = df_data[TARGET].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train_s, X_test_s = standardize(X_train, X_test)

    processor = PostProcessor()
    rows: List[Dict[str, Any]] = []

    for name, predict_fn, proba_fn in build_rows(quick=quick):
        y_pred = predict_fn(X_train_s, y_train, X_test_s)
        metrics = processor.classification_metrics(y_test, y_pred, average="binary")
        acc = processor.accuracy(y_test, y_pred)
        auc: Optional[float] = None
        if proba_fn is not None:
            proba = proba_fn(X_train_s, y_train, X_test_s)
            auc = _safe_roc_auc(y_test, proba)

        rows.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(float(metrics["precision"]), 4),
                "Recall": round(float(metrics["recall"]), 4),
                "F1": round(float(metrics["f1_score"]), 4),
                "ROC-AUC": None if auc is None else round(auc, 4),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["dataset_rows"] = len(df_data)
    out.attrs["features"] = list(FEATURES)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rice_ml models on the spam email CSV.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter training (fewer epochs / fewer trees) for a faster smoke run.",
    )
    args = parser.parse_args()

    df_result = run_comparison(quick=args.quick)
    n = int(df_result.attrs.get("dataset_rows", 0))
    print(f"Dataset: spam_email_dataset.csv ({n} rows)")
    print(f"Features: {FEATURES}")
    print(f"Train/test split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    print(f"Standardization: training-set mean/std applied to train and test.")
    print()

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    table = []
    for _, row in df_result.iterrows():
        table.append(
            [
                row["Model"],
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1']:.4f}",
                "" if pd.isna(row["ROC-AUC"]) else f"{row['ROC-AUC']:.4f}",
            ]
        )

    if tabulate is not None:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        col_widths = [max(len(headers[j]), max(len(str(row[j])) for row in table)) for j in range(len(headers))]
        fmt = "  ".join("{:%ds}" % w for w in col_widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_widths))
        for row in table:
            print(fmt.format(*row))


if __name__ == "__main__":
    main()
