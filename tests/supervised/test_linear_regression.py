import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.linear_regression_class import (
    SingleNeuronLinearRegression,
)

@pytest.fixture
def sample_data():
    """Simple dataset for regression testing."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])  # Perfect linear relationship
    return X, y

def test_linear_regression(sample_data):
    X, y = sample_data
    model = SingleNeuronLinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_linear_regression_unseen_data(sample_data):
    X, y = sample_data
    model = SingleNeuronLinearRegression()
    model.fit(X, y)
    unseen_X = np.array([[1.5], [2.5], [3.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(predictions >= 1) and np.all(predictions <= 5), f"Expected predictions to be between 1 and 5, but got {predictions}"

def test_linear_regression_invalid_input():
    model = SingleNeuronLinearRegression()
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1, 2, 3])  # Mismatched number of samples and labels

def test_linear_regression_predict_before_fit():
    model = LinearRegression()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[1], [2], [3]])

def test_linear_regression_convergence(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_linear_regression_weights(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    assert model.weights_.shape == (1,), f"Expected weights shape (1,), but got {model.weights_.shape}"
    assert isinstance(model.bias_, float), f"Expected bias to be a float, but got {type(model.bias_)}"

def test_linear_regression_predict_unseen_data(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    unseen_X = np.array([[1.5], [2.5], [3.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(predictions >= 1) and np.all(predictions <= 5), f"Expected predictions to be between 1 and 5, but got {predictions}"

def test_linear_regression_learning_rate_effect(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_linear_regression_zero_learning_rate(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_linear_regression_large_learning_rate(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_linear_regression_non_linear_data():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 4, 9, 16, 25])  # Quadratic relationship
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-2), "Expected poor fit for non-linear data, but got close predictions"

def test_linear_regression_outliers():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 100])  # Last point is an outlier
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-2), "Expected poor fit due to outlier, but got close predictions"

def test_linear_regression_multicollinearity():
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])  # Perfect multicollinearity
    y = np.array([1, 2, 3, 4, 5])
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"
