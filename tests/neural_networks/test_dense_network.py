import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.Neural_Networks.dense_network_class import DenseNetwork

@pytest.fixture

def sample_data():
    """Simple dataset for testing DenseNetwork."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR logic gate
    return X, y

def test_dense_network(sample_data):
    X, y = sample_data
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.1, max_iter=1000, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_dense_network_unseen_data(sample_data):
    X, y = sample_data
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.1, max_iter=1000, random_state=42)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3, 1), f"Expected predictions shape (3, 1), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_dense_network_invalid_input():
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.1, max_iter=1000)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [[0]])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [[0], [1], [1]])  # Mismatched number of samples and labels

def test_dense_network_predict_before_fit():
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.1, max_iter=1000)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])

def test_dense_network_convergence(sample_data):
    X, y = sample_data
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.1, max_iter=1000, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_dense_network_weights(sample_data):
    X, y = sample_data
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.1, max_iter=1000, random_state=42)
    model.fit(X, y)
    assert len(model.coefs_) == 2, f"Expected 2 layers of weights, but got {len(model.coefs_)}"
    assert model.coefs_[0].shape == (2, 2), f"Expected first layer weights shape (2, 2), but got {model.coefs_[0].shape}"
    assert model.coefs_[1].shape == (2, 1), f"Expected second layer weights shape (2, 1), but got {model.coefs_[1].shape}"

def test_dense_network_learning_rate_effect(sample_data):
    X, y = sample_data
    model_low_lr = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.01, max_iter=1000, random_state=42)
    model_low_lr.fit(X, y)
    predictions_low_lr = model_low_lr.predict(X)
    
    model_high_lr = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=1.0, max_iter=1000, random_state=42)
    model_high_lr.fit(X, y)
    predictions_high_lr = model_high_lr.predict(X)
    
    assert np.array_equal(predictions_low_lr, y), f"Expected {y} with low learning rate, but got {predictions_low_lr}"
    assert not np.array_equal(predictions_high_lr, y), f"Expected different predictions with high learning rate, but got {predictions_high_lr}"

def test_dense_network_zero_learning_rate(sample_data):
    X, y = sample_data
    model = DenseNetwork(hidden_layer_sizes=(2,), learning_rate=0.0, max_iter=1000, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With zero learning rate, predictions should not match labels"
