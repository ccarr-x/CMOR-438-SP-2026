import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Unsupervised_Learning.pca import PCA


@pytest.fixture
def sample_data():
    """Simple 2D dataset with clear variance structure."""
    return np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=float
    )


@pytest.fixture
def high_dim_data():
    """Higher-dimensional dataset for testing."""
    np.random.seed(42)
    return np.random.randn(20, 10)


def test_pca_fit_basic(sample_data):
    model = PCA(n_components=1)
    model.fit(sample_data)

    assert model.components_.shape == (1, 2)
    assert model.explained_variance_.shape == (1,)
    assert model.explained_variance_ratio_.shape == (1,)
    assert len(model.mean_) == 2
    assert model.n_features_ == 2


def test_pca_fit_all_components(sample_data):
    model = PCA(n_components=None)
    model.fit(sample_data)

    assert model.components_.shape == (2, 2)
    assert model.explained_variance_.shape == (2,)
    assert np.sum(model.explained_variance_ratio_) <= 1.01  # Account for floating point


def test_pca_transform(sample_data):
    model = PCA(n_components=1)
    model.fit(sample_data)
    transformed = model.transform(sample_data)

    assert transformed.shape == (4, 1)
    assert np.all(np.isfinite(transformed))


def test_pca_fit_transform(sample_data):
    model = PCA(n_components=1)
    transformed = model.fit_transform(sample_data)

    assert transformed.shape == (4, 1)
    assert np.all(np.isfinite(transformed))


def test_pca_inverse_transform(sample_data):
    model = PCA(n_components=1)
    model.fit(sample_data)
    transformed = model.transform(sample_data)
    reconstructed = model.inverse_transform(transformed)

    assert reconstructed.shape == (4, 2)
    assert np.all(np.isfinite(reconstructed))


def test_pca_reconstruction_error(sample_data):
    """Test that reconstruction error is small for full components."""
    model = PCA(n_components=None)
    transformed = model.fit_transform(sample_data)
    reconstructed = model.inverse_transform(transformed)

    np.testing.assert_array_almost_equal(reconstructed, sample_data, decimal=5)


def test_pca_explained_variance_decreasing():
    """Test that explained variance ratios are in decreasing order."""
    np.random.seed(42)
    X = np.random.randn(30, 5)
    model = PCA(n_components=None)
    model.fit(X)

    assert np.all(
        np.diff(model.explained_variance_ratio_) <= 0
    ), "Explained variance should be decreasing"


def test_pca_svd_vs_covariance(sample_data):
    """Test that SVD and covariance methods give similar results."""
    model_cov = PCA(n_components=1, use_svd=False)
    model_svd = PCA(n_components=1, use_svd=True)

    model_cov.fit(sample_data)
    model_svd.fit(sample_data)

    transformed_cov = model_cov.transform(sample_data)
    transformed_svd = model_svd.transform(sample_data)

    np.testing.assert_array_almost_equal(
        np.abs(transformed_cov), np.abs(transformed_svd), decimal=5
    )


def test_pca_transform_raises_before_fit(sample_data):
    model = PCA(n_components=1)
    with pytest.raises(ValueError, match="not fitted"):
        model.transform(sample_data)


def test_pca_inverse_transform_raises_before_fit(sample_data):
    model = PCA(n_components=1)
    with pytest.raises(ValueError, match="not fitted"):
        model.inverse_transform(np.random.randn(4, 1))


def test_pca_invalid_input_dim(sample_data):
    model = PCA(n_components=1)
    model.fit(sample_data)

    with pytest.raises(ValueError, match="two-dimensional"):
        model.transform(np.array([1.0, 2.0]))


def test_pca_feature_mismatch(sample_data):
    model = PCA(n_components=1)
    model.fit(sample_data)

    with pytest.raises(ValueError, match="features"):
        model.transform(np.random.randn(4, 3))


def test_pca_n_components_exceeds_features(high_dim_data):
    model = PCA(n_components=15)
    with pytest.raises(ValueError, match="cannot be greater"):
        model.fit(high_dim_data)


def test_pca_invalid_n_components():
    with pytest.raises(ValueError, match="must be at least 1"):
        PCA(n_components=0)


def test_pca_invalid_use_svd():
    with pytest.raises(ValueError, match="must be a boolean"):
        PCA(use_svd="yes")
