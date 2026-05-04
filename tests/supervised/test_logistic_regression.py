import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.logistic_regression_class import LogisticRegression

@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate
    return X, y

def test_logistic_regression(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_logistic_regression_unseen_data(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_logistic_regression_invalid_input():
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels

def test_logistic_regression_predict_before_fit():
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])

def test_logistic_regression_convergence(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_logistic_regression_non_convergence(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=10.0, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With large learning rate, model may diverge and predictions should not match labels"

def test_logistic_regression_weights(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    assert model.weights_.shape == (2,), f"Expected weights shape (2,), but got {model.weights_.shape}"
    assert isinstance(model.bias_, float), f"Expected bias to be a float, but got {type(model.bias_)}"

def test_logistic_regression_predict_unseen_data(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_logistic_regression_non_linearly_separable():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR logic gate
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "Logistic Regression should not be able to learn XOR function"

def test_logistic_regression_learning_rate_effect(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=1.0, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_logistic_regression_zero_learning_rate(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=0.0, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With zero learning rate, model should not learn and predictions should not match labels"

def test_logistic_regression_large_learning_rate(sample_data):
    X, y = sample_data
    model = LogisticRegression(learning_rate=10.0, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "With large learning rate, model may diverge and predictions should not match labels"

def test_logistic_regression_non_linear_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR logic gate
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y), "Logistic Regression should not be able to learn non-linear data like XOR function"

def test_logistic_regression_all_zero_labels():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 0])  # All labels are the same
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_logistic_regression_all_one_labels():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 1, 1, 1])  # All labels are the same
    model = LogisticRegression(learning_rate=0.1, n_iters=100)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"
