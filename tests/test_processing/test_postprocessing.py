import numpy as np
import pytest

from rice_ml.processing.postprocessing import PostProcessor


@pytest.fixture
def processor():
    return PostProcessor()


def test_classification_metrics_binary(processor):
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    out = processor.classification_metrics(y_true, y_pred, average="binary")
    assert "classification_error" in out
    assert "precision" in out
    assert "recall" in out
    assert "f1_score" in out
    assert out["confusion_matrix"].shape == (2, 2)
    assert 0.0 <= out["classification_error"] <= 1.0


def test_precision_recall_f1_helpers(processor):
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    p = processor.precision(y_true, y_pred, average="binary")
    r = processor.recall(y_true, y_pred, average="binary")
    f = processor.f1_score(y_true, y_pred, average="binary")
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f <= 1.0


def test_accuracy(processor):
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])
    assert processor.accuracy(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_multiclass_average_macro(processor):
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 2, 2, 2, 1])
    out = processor.classification_metrics(y_true, y_pred, average="macro")
    assert out["confusion_matrix"].shape == (3, 3)


def test_binary_requires_two_classes(processor):
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        processor.precision(y_true, y_pred, average="binary")
