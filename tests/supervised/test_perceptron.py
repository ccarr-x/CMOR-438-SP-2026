import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.perceptron_class import Perceptron

@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate
    return X, y

def test_perceptron(sample_data):
    """Test basic perceptron training and prediction on linearly separable data."""
    X, y = sample_data
    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_perceptron_non_linearly_separable():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR logic gate
    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "Perceptron should not be able to learn XOR function"    

def test_perceptron_convergence(sample_data):
    X, y = sample_data
    model = Perceptron(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_perceptron_weights(sample_data):
    X, y = sample_data
    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    assert model.weights_.shape == (2,), f"Expected weights shape (2,), but got {model.weights_.shape}"
    assert isinstance(model.bias_, float), f"Expected bias to be a float, but got {type(model.bias_)}"

def test_perceptron_predict_unseen_data(sample_data):
    X, y = sample_data
    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_perceptron_invalid_input():
    model = Perceptron(learning_rate=0.1, n_iters=10)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.predict([[0.5, 0.5], [0.2, 0.8]])  # Predict before fitting

def test_perceptron_learning_rate_effect(sample_data):
    X, y = sample_data
    model = Perceptron(learning_rate=1.0, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_perceptron_zero_learning_rate(sample_data):
    X, y = sample_data
    model = Perceptron(learning_rate=0.0, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With zero learning rate, predictions should not match labels"

def test_perceptron_negative_learning_rate(sample_data):
    X, y = sample_data
    model = Perceptron(learning_rate=-0.1, n_iters=10)
    with pytest.raises(ValueError):
        model.fit(X, y)
        