import numpy as np
import pytest

from rice_ml.Neural_Networks.dense_network_class import DenseNetwork


@pytest.fixture
def xor_data():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    return X, y


def test_dense_train_predict_shapes(xor_data):
    X, y = xor_data
    model = DenseNetwork(hidden_layer_sizes=(8,), eta=0.1, epochs=400, random_state=42)
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (4,)
    assert preds.dtype == float or preds.dtype == np.float64


def test_predict_before_train_raises():
    model = DenseNetwork(hidden_layer_sizes=(4,), eta=0.1, epochs=10, random_state=0)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.array([[0.0, 1.0]]))


def test_train_mismatched_lengths_raises(xor_data):
    X, _ = xor_data
    model = DenseNetwork(hidden_layer_sizes=(4,), eta=0.1, epochs=10, random_state=0)
    with pytest.raises(ValueError, match="one entry per row"):
        model.train(X, np.array([0.0]))


def test_weights_populated_after_train(xor_data):
    X, y = xor_data
    model = DenseNetwork(hidden_layer_sizes=(4,), eta=0.1, epochs=50, random_state=1)
    model.train(X, y)
    assert len(model.weights_) >= 2
    assert len(model.biases_) >= 2
    assert len(model.loss_history_) > 0
