import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.linear_decision_tree_regressor import DecisionTreeRegressor

@pytest.fixture
def sample_data():
    """Simple dataset for regression testing."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])  # Perfect linear relationship
    return X, y

def test_decision_tree_regressor(sample_data):
    X, y = sample_data
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_decision_tree_regressor_unseen_data(sample_data):
    X, y = sample_data
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    unseen_X = np.array([[1.5], [2.5], [3.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(predictions >= 1) and np.all(predictions <= 5), f"Expected predictions to be between 1 and 5, but got {predictions}"

def test_decision_tree_regressor_overfitting(sample_data):
    X, y = sample_data
    model = DecisionTreeRegressor(max_depth=1)  # Very shallow tree to force underfitting
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "Expected underfitting with max_depth=1, but got perfect predictions"

def test_decision_tree_regressor_invalid_input():
    model = DecisionTreeRegressor(max_depth=3)
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1, 2, 3])  # Mismatched number of samples and labels

def test_decision_tree_regressor_predict_before_fit():
    model = DecisionTreeRegressor(max_depth=3)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[1], [2], [3]])

def test_decision_tree_regressor_max_depth(sample_data):
    X, y = sample_data
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (5,), f"Expected predictions shape (5,), but got {predictions.shape}"
    assert np.all(predictions >= 1) and np.all(predictions <= 5), f"Expected predictions to be between 1 and 5, but got {predictions}"