import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent / "src"))

from rice_ml.Unsupervised_Learning.k_means_clustering import KMeans


@pytest.fixture
def sample_data():
    return np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
    )


def test_fit_predict_simple(sample_data):
    model = KMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(sample_data)

    assert labels.shape == (4,)
    assert model.cluster_centers_.shape == (2, 2)
    assert model.n_iter_ > 0

    expected_labels = np.argmin(
        np.linalg.norm(sample_data[:, np.newaxis] - model.cluster_centers_, axis=2), axis=1
    )
    np.testing.assert_array_equal(labels, expected_labels)


def test_fit_transform_returns_full_distance_matrix(sample_data):
    model = KMeans(n_clusters=2, random_state=42)
    transformed = model.fit_transform(sample_data)

    assert transformed.shape == (4, 2)
    assert np.all(transformed >= 0.0)
    assert np.allclose(
        transformed,
        np.linalg.norm(sample_data[:, np.newaxis] - model.cluster_centers_, axis=2),
    )


def test_predict_raises_before_fit():
    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.array([[0.0, 0.0]], dtype=float))


def test_n_clusters_exceeds_samples_raises():
    model = KMeans(n_clusters=5)
    with pytest.raises(
        ValueError,
        match=r"n_clusters \(5\) cannot exceed n_samples \(4\)\.",
    ):
        model.fit(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                ],
                dtype=float,
            )
        )
