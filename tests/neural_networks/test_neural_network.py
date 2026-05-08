import numpy as np
import pytest

from rice_ml.Neural_Networks.neural_network_class import SigmoidNeuralNetwork


@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def test_sigmoid_nn_train_predict_shapes(sample_data):
    X, y = sample_data
    model = SigmoidNeuralNetwork(
        hidden_layer_sizes=(8,),
        eta=0.5,
        epochs=200,
        random_state=42,
    )
    model.train(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (4,)
    assert np.all(np.isin(predictions, [0, 1]))


def test_train_mismatched_lengths_raises(sample_data):
    X, _ = sample_data
    model = SigmoidNeuralNetwork(hidden_layer_sizes=(4,), eta=0.1, epochs=10, random_state=0)
    with pytest.raises(ValueError, match="one entry per row"):
        model.train(X, np.array([0, 1]))


def test_train_invalid_x_ndim():
    model = SigmoidNeuralNetwork(hidden_layer_sizes=(4,), eta=0.1, epochs=10, random_state=0)
    with pytest.raises(ValueError, match="two-dimensional"):
        model.train(np.array([1, 2, 3]), np.array([0, 1, 0]))


def test_predict_proba_after_train(sample_data):
    X, y = sample_data
    model = SigmoidNeuralNetwork(hidden_layer_sizes=(4,), eta=0.3, epochs=120, random_state=1)
    model.train(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (4,)
    assert np.all((probs >= 0) & (probs <= 1))
