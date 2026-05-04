import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.decision_tree_class import DecisionTree

@pytest.fixture
def sample_data():
    """Simple dataset for classification testing."""
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1, 1])  # Binary classes
    return X, y

def test_decision_tree(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_decision_tree_unseen_data(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=3)
    model.fit(X, y)
    unseen_X = np.array([[0.5], [1.5], [2.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_decision_tree_overfitting(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=1)  # Very shallow tree to force underfitting
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "Expected underfitting with max_depth=1, but got perfect predictions"

def test_decision_tree_invalid_input():
    model = DecisionTree(max_depth=3)
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0, 1, 1])  # Mismatched number of samples and labels

def test_decision_tree_predict_before_fit():
    model = DecisionTree(max_depth=3)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0], [1], [2]])

def test_decision_tree_max_depth(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (5,), f"Expected predictions shape (5,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_decision_tree_non_binary_classes():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 1, 2, 3, 4])  # Multi-class labels
    model = DecisionTree(max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_decision_tree_depth_effect(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=1)  # Shallow tree
    model.fit(X, y)
    predictions_shallow = model.predict(X)

    model_deep = DecisionTree(max_depth=10)  # Deep tree
    model_deep.fit(X, y)
    predictions_deep = model_deep.predict(X)

    assert not np.array_equal(predictions_shallow, y), "Expected underfitting with max_depth=1, but got perfect predictions"
    assert np.array_equal(predictions_deep, y), f"Expected {y}, but got {predictions_deep}"

def test_decision_tree_predict_unseen_data(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=3)
    model.fit(X, y)
    unseen_X = np.array([[0.5], [1.5], [2.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_decision_tree_invalid_input():
    model = DecisionTree(max_depth=3)
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0, 1, 1])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.predict([[0.5], [1.5], [2.5]])  # Predict before fitting

def test_decision_tree_max_depth_effect(sample_data):
    X, y = sample_data
    model = DecisionTree(max_depth=1)  # Shallow tree
    model.fit(X, y)
    predictions_shallow = model.predict(X)

    model_deep = DecisionTree(max_depth=10)  # Deep tree
    model_deep.fit(X, y)
    predictions_deep = model_deep.predict(X)

    assert not np.array_equal(predictions_shallow, y), "Expected underfitting with max_depth=1, but got perfect predictions"
    assert np.array_equal(predictions_deep, y), f"Expected {y}, but got {predictions_deep}"