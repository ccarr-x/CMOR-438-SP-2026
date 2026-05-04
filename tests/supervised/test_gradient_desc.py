import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.gradient_desc_class import GradientDescent

@pytest.fixture
def sample_data():
    """Simple dataset for regression testing."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])  # Perfect linear relationship
    return X, y

def test_gradient_descent(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_gradient_descent_unseen_data(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    unseen_X = np.array([[1.5], [2.5], [3.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(predictions >= 1) and np.all(predictions <= 5), f"Expected predictions to be between 1 and 5, but got {predictions}"

def test_gradient_descent_invalid_input():
    model = GradientDescent(learning_rate=0.01, n_iters=1000)
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1, 2, 3])  # Mismatched number of samples and labels

def test_gradient_descent_predict_before_fit():
    model = GradientDescent(learning_rate=0.01, n_iters=1000)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[1], [2], [3]])

def test_gradient_descent_convergence(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.01, n_iters=10000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_gradient_descent_weights(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    assert model.weights_.shape == (1,), f"Expected weights shape (1,), but got {model.weights_.shape}"
    assert isinstance(model.bias_, float), f"Expected bias to be a float, but got {type(model.bias_)}"

def test_gradient_descent_predict_unseen_data(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    unseen_X = np.array([[1.5], [2.5], [3.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(predictions >= 1) and np.all(predictions <= 5), f"Expected predictions to be between 1 and 5, but got {predictions}"

def test_gradient_descent_learning_rate_effect(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y, atol=1e-2), f"Expected {y}, but got {predictions}"

def test_gradient_descent_zero_learning_rate(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.0, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-2), "Expected no learning with zero learning rate, but got close predictions"

def test_gradient_descent_large_learning_rate(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=1.0, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-2), "Expected divergence with large learning rate, but got close predictions"

def test_gradient_descent_non_convergence(sample_data):
    X, y = sample_data
    model = GradientDescent(learning_rate=0.01, n_iters=10)  # Too few iterations to converge
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.allclose(predictions, y, atol=1e-2), "Expected non-convergence with too few iterations, but got close predictions"