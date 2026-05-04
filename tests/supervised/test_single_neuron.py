import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.single_neuron_model_class import SingleNeuronModel

@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate
    return X, y

def test_single_neuron_model(sample_data):
    X, y = sample_data
    model = SingleNeuronModel(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_single_neuron_model_non_linearly_separable():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR logic gate
    model = SingleNeuronModel(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "Single neuron model should not be able to learn XOR function"

def test_single_neuron_model_convergence(sample_data):
    X, y = sample_data
    model = SingleNeuronModel(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_single_neuron_model_weights(sample_data):
    X, y = sample_data
    model = SingleNeuronModel(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    assert model.weights_.shape == (2,), f"Expected weights shape (2,), but got {model.weights_.shape}"
    assert isinstance(model.bias_, float), f"Expected bias to be a float, but got {type(model.bias_)}"

def test_single_neuron_model_predict_unseen_data(sample_data):
    X, y = sample_data
    model = SingleNeuronModel(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_single_neuron_model_invalid_input():
    model = SingleNeuronModel(learning_rate=0.1, n_iters=10)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.predict([[0.5, 0.5], [0.2, 0.8]])  # Predict before fitting

def test_single_neuron_model_zero_learning_rate(sample_data):
    X, y = sample_data
    model = SingleNeuronModel(learning_rate=0.0, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, np.zeros_like(y)), f"Expected all predictions to be 0 due to zero learning rate, but got {predictions}"

def test_single_neuron_model_large_learning_rate(sample_data):
    X, y = sample_data
    model = SingleNeuronModel(learning_rate=10.0, n_iters=10)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"