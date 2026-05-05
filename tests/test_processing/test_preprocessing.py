import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.processing.preprocessing import train_test_split, standardize, normalize, min_max_scale, encode_labels, transform_labels

@pytest.fixture

def sample_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y

def test_train_test_split(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    assert X_train.shape == (3, 2), f"Expected X_train shape (3, 2), but got {X_train.shape}"
    assert X_test.shape == (1, 2), f"Expected X_test shape (1, 2), but got {X_test.shape}"
    assert y_train.shape == (3,), f"Expected y_train shape (3,), but got {y_train.shape}"
    assert y_test.shape == (1,), f"Expected y_test shape (1,), but got {y_test.shape}"
    assert np.array_equal(X_train[0], [5, 6]), f"Expected first training sample to be [5, 6], but got {X_train[0]}"
    assert np.array_equal(y_train[0], 0), f"Expected first training label to be 0, but got {y_train[0]}"

def test_standardize(sample_data):
    X, _ = sample_data
    X_std = standardize(X)
    assert np.allclose(X_std.mean(axis=0), 0), f"Expected mean to be approximately 0, but got {X_std.mean(axis=0)}"
    assert np.allclose(X_std.std(axis=0), 1), f"Expected std to be approximately 1, but got {X_std.std(axis=0)}"

def test_normalize(sample_data):
    X, _ = sample_data
    X_norm = normalize(X)
    norms = np.linalg.norm(X_norm, axis=1)
    assert np.allclose(norms, 1), f"Expected all samples to have norm 1, but got {norms}"

def test_min_max_scale(sample_data):
    X, _ = sample_data
    X_scaled = min_max_scale(X)
    assert np.allclose(X_scaled.min(axis=0), 0), f"Expected min to be approximately 0, but got {X_scaled.min(axis=0)}"
    assert np.allclose(X_scaled.max(axis=0), 1), f"Expected max to be approximately 1, but got {X_scaled.max(axis=0)}"

def test_encode_labels(sample_data):
    _, y = sample_data
    encoded = encode_labels(y)
    assert encoded.shape == (4, 2), f"Expected encoded labels shape (4, 2), but got {encoded.shape}"
    assert np.array_equal(encoded[0], [1, 0]), f"Expected first encoded label to be [1, 0], but got {encoded[0]}"
    assert np.array_equal(encoded[1], [0, 1]), f"Expected second encoded label to be [0, 1], but got {encoded[1]}"

def test_transform_labels(sample_data):
    _, y = sample_data
    transformed = transform_labels(y, {0: 'A', 1: 'B'})
    assert transformed.shape == (4,), f"Expected transformed labels shape (4,), but got {transformed.shape}"
    assert np.array_equal(transformed, ['A', 'B', 'A', 'B']), f"Expected transformed labels to be ['A', 'B', 'A', 'B'], but got {transformed}"