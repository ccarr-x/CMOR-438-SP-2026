import numpy as np
import pytest

pytest.importorskip("tensorflow")

from rice_ml.Neural_Networks.preprocessing import flatten_images, one_hot_encode_labels


@pytest.fixture
def sample_image_data():
    images = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.float32)
    labels = np.array([0, 1])
    return images, labels


def test_flatten_images(sample_image_data):
    images, _ = sample_image_data
    flattened = flatten_images(images)
    assert flattened.shape == (2, 4)
    assert np.allclose(flattened[0], [0, 1, 1, 0])
    assert np.allclose(flattened[1], [1, 0, 0, 1])


def test_one_hot_encode_labels(sample_image_data):
    _, labels = sample_image_data
    encoded = one_hot_encode_labels(labels, num_classes=2)
    assert encoded.shape == (2, 2)
    assert np.allclose(encoded[0], [1, 0])
    assert np.allclose(encoded[1], [0, 1])


def test_one_hot_encode_invalid_num_classes(sample_image_data):
    _, labels = sample_image_data
    with pytest.raises(ValueError):
        one_hot_encode_labels(labels, num_classes=1)
