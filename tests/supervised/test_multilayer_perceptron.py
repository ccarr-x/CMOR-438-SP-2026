import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.multilayer_perceptron_class import MultilayerPerceptron

@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR logic gate
    return X, y

def test_multilayer_perceptron(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_multilayer_perceptron_unseen_data(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_multilayer_perceptron_invalid_input():
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels

def test_multilayer_perceptron_predict_before_fit():
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])

def test_multilayer_perceptron_convergence(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=10000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_multilayer_perceptron_weights(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    assert len(model.weights_) == 2, f"Expected 2 layers of weights, but got {len(model.weights_)}"
    assert model.weights_[0].shape == (2, 5), f"Expected first layer weights shape (2, 5), but got {model.weights_[0].shape}"
    assert model.weights_[1].shape == (5, 1), f"Expected second layer weights shape (5, 1), but got {model.weights_[1].shape}"
    assert isinstance(model.biases_[0], np.ndarray) and model.biases_[0].shape == (5,), f"Expected first layer biases shape (5,), but got {model.biases_[0].shape}"
    assert isinstance(model.biases_[1], np.ndarray) and model.biases_[1].shape == (1,), f"Expected second layer biases shape (1,), but got {model.biases_[1].shape}"

def test_multilayer_perceptron_predict_unseen_data(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_multilayer_perceptron_learning_rate_effect(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=1.0, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_multilayer_perceptron_zero_learning_rate(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.0, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With zero learning rate, model should not learn and predictions should not match labels"

def test_multilayer_perceptron_non_convergence(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=10.0, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With large learning rate, model may diverge and predictions should not match labels"

def test_multilayer_perceptron_non_linearly_separable():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR logic gate
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_multilayer_perceptron_large_hidden_layer(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(100,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_multilayer_perceptron_multiple_hidden_layers(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5, 5), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_multilayer_perceptron_predict_unseen_data(sample_data):
    X, y = sample_data
    model = MultilayerPerceptron(hidden_layer_sizes=(5,), learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"