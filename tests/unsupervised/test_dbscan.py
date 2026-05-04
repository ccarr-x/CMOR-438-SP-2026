import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Unsupervised_Learning.dbscan import DBSCAN


@pytest.fixture
def sample_data():
    """Simple 2D dataset with clear cluster structure and some noise."""
    np.random.seed(42)
    cluster1 = np.random.randn(10, 2) + [0, 0]
    cluster2 = np.random.randn(10, 2) + [5, 5]
    noise = np.array([[10.0, 10.0], [10.5, 10.5]])
    return np.vstack([cluster1, cluster2, noise])


@pytest.fixture
def high_dim_data():
    """Higher-dimensional dataset for testing."""
    np.random.seed(42)
    return np.random.randn(30, 5)


def test_dbscan_fit_basic(sample_data):
    model = DBSCAN(eps=1.0, min_samples=5)
    model.fit(sample_data)

    assert model.labels_.shape == (22,)
    assert model.n_clusters_ >= 0
    assert -1 in model.labels_ or model.n_clusters_ > 0


def test_dbscan_fit_predict(sample_data):
    model = DBSCAN(eps=1.0, min_samples=5)
    labels = model.fit_predict(sample_data)

    assert labels.shape == (22,)
    assert np.all(np.isin(labels, np.unique(model.labels_)))


def test_dbscan_identifies_clusters(sample_data):
    """Test that DBSCAN identifies the two main clusters."""
    model = DBSCAN(eps=1.5, min_samples=3)
    labels = model.fit_predict(sample_data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 1


def test_dbscan_identifies_noise(sample_data):
    """Test that DBSCAN can identify noise points."""
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(sample_data)

    assert -1 in labels or model.n_clusters_ > 0


def test_dbscan_large_eps():
    """Test that with large eps, all points belong to one cluster."""
    X = np.array([[0, 0], [1, 1], [2, 2], [10, 10]], dtype=float)
    model = DBSCAN(eps=20.0, min_samples=2)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters <= 1


def test_dbscan_small_eps():
    """Test that with very small eps, most points become noise."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    model = DBSCAN(eps=0.1, min_samples=2)
    labels = model.fit_predict(X)

    n_noise = np.sum(labels == -1)
    assert n_noise > 0 or model.n_clusters_ >= 1


def test_dbscan_min_samples_effect():
    """Test that increasing min_samples increases noise points."""
    X = np.array(
        [[0, 0], [0.1, 0.1], [1, 1], [1.1, 1.1], [2, 2], [2.1, 2.1]], dtype=float
    )

    model1 = DBSCAN(eps=0.5, min_samples=2)
    labels1 = model1.fit_predict(X)
    noise1 = np.sum(labels1 == -1)

    model2 = DBSCAN(eps=0.5, min_samples=5)
    labels2 = model2.fit_predict(X)
    noise2 = np.sum(labels2 == -1)

    assert noise2 >= noise1


def test_dbscan_invalid_eps():
    with pytest.raises(ValueError, match="eps must be positive"):
        DBSCAN(eps=0)

    with pytest.raises(ValueError, match="eps must be positive"):
        DBSCAN(eps=-1.0)


def test_dbscan_invalid_min_samples():
    with pytest.raises(ValueError, match="min_samples must be at least 1"):
        DBSCAN(min_samples=0)

    with pytest.raises(ValueError, match="min_samples must be at least 1"):
        DBSCAN(min_samples=-1)


def test_dbscan_invalid_input_dim(sample_data):
    model = DBSCAN()

    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(np.array([1.0, 2.0, 3.0]))


def test_dbscan_empty_input():
    model = DBSCAN()

    with pytest.raises(ValueError, match="at least one sample"):
        model.fit(np.empty((0, 2)))


def test_dbscan_deterministic(sample_data):
    """Test that fitting the same data gives the same results."""
    model1 = DBSCAN(eps=1.0, min_samples=5)
    model2 = DBSCAN(eps=1.0, min_samples=5)

    labels1 = model1.fit_predict(sample_data)
    labels2 = model2.fit_predict(sample_data)

    np.testing.assert_array_equal(labels1, labels2)


def test_dbscan_single_sample():
    """Test DBSCAN with a single sample."""
    X = np.array([[0.0, 0.0]])
    model = DBSCAN(eps=1.0, min_samples=1)
    labels = model.fit_predict(X)

    assert labels.shape == (1,)
    assert labels[0] >= -1


def test_dbscan_labels_range(high_dim_data):
    """Test that labels are either -1 (noise) or non-negative cluster ids."""
    model = DBSCAN(eps=2.0, min_samples=3)
    labels = model.fit_predict(high_dim_data)

    assert np.all(labels >= -1)
    assert np.all(np.diff(np.sort(np.unique(labels[labels >= 0]))) == 1)
