import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.linear_knn_class import KNN

@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate
    return X, y

def test_knn(sample_data):
    X, y = sample_data
    model = KNN(n_neighbors=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_knn_unseen_data(sample_data):
    X, y = sample_data
    model = KNN(n_neighbors=3)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_knn_invalid_input():
    model = KNN(n_neighbors=3)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels

def test_knn_predict_before_fit():
    model = KNN(n_neighbors=3)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])

def test_knn_k_greater_than_samples(sample_data):
    X, y = sample_data
    model = KNN(n_neighbors=5)  # More neighbors than samples
    with pytest.raises(ValueError, match=r"n_neighbors \(5\) cannot be greater than n_samples \(4\)\."):
        model.fit(X, y)

def test_knn_k_equals_samples(sample_data):
    X, y = sample_data
    model = KNN(n_neighbors=4)  # k equals number of samples
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_knn_k_equals_(sample_data):
    X, y = sample_data
    equals = [1,2,3,4]
    for k in equals:
        model = KNN(n_neighbors=k)  # test k equals 1-4
        model.fit(X, y)
        predictions = model.predict(X)
        assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"