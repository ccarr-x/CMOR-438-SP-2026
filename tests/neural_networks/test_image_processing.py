import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.Neural_Networks.preprocessing import flatten_images, one_hot_encode_labels

@pytest.fixture

def sample_image_data():
    """Simple dataset of 2x2 images for testing."""
    images = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])  # Two 2x2 images
    labels = np.array([0, 1])  # Corresponding labels
    return images, labels

def test_flatten_images(sample_image_data):
    images, _ = sample_image_data
    flattened = flatten_images(images)
    expected_shape = (2, 4)  # 2 images, each flattened to 4 pixels
    assert flattened.shape == expected_shape, f"Expected shape {expected_shape}, but got {flattened.shape}"
    assert np.array_equal(flattened[0], [0, 1, 1, 0]), f"Expected first image to be flattened to [0, 1, 1, 0], but got {flattened[0]}"
    assert np.array_equal(flattened[1], [1, 0, 0, 1]), f"Expected second image to be flattened to [1, 0, 0, 1], but got {flattened[1]}"

def test_one_hot_encode_labels(sample_image_data):
    _, labels = sample_image_data
    encoded = one_hot_encode_labels(labels, num_classes=2)
    expected_shape = (2, 2)  # 2 samples, 2 classes
    assert encoded.shape == expected_shape, f"Expected shape {expected_shape}, but got {encoded.shape}"
    assert np.array_equal(encoded[0], [1, 0]), f"Expected first label to be one-hot encoded as [1, 0], but got {encoded[0]}"
    assert np.array_equal(encoded[1], [0, 1]), f"Expected second label to be one-hot encoded as [0, 1], but got {encoded[1]}"

def test_one_hot_encode_labels_invalid_input(sample_image_data):
    _, labels = sample_image_data
    with pytest.raises(ValueError):
        one_hot_encode_labels(labels, num_classes=1)  # num_classes must be greater than max label
    with pytest.raises(ValueError):
        one_hot_encode_labels(labels, num_classes=0)  # num_classes must be positive
    with pytest.raises(ValueError):
        one_hot_encode_labels(labels, num_classes=-1)  # num_classes must be positive
    with pytest.raises(ValueError):
        one_hot_encode_labels(labels, num_classes=1.5)  # num_classes must be an integer

def test_flatten_images_invalid_input(sample_image_data):
    images, _ = sample_image_data
    with pytest.raises(ValueError):
        flatten_images([1, 2, 3])  # Not a numpy array
    with pytest.raises(ValueError):
        flatten_images(np.array([1, 2, 3]))  # Not a 3D array of images
    with pytest.raises(ValueError):
        flatten_images(np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [0, 0]]]))  # More than 2 images in the batch
    with pytest.raises(ValueError):
        flatten_images(np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]), expected_image_shape=(3, 3))  # Mismatched expected image shape

def test_one_hot_encode_labels_non_integer_labels(sample_image_data):
    _, labels = sample_image_data
    with pytest.raises(ValueError):
        one_hot_encode_labels([0.5, 1.5], num_classes=2)  # Non-integer labels
    with pytest.raises(ValueError):
        one_hot_encode_labels([-1, 0], num_classes=2)  # Negative label value
    with pytest.raises(ValueError):
        one_hot_encode_labels([0, 2], num_classes=2)  # Label value equal to num_classes
    with pytest.raises(ValueError):
        one_hot_encode_labels([0, 3], num_classes=2)  # Label value greater than num_classes