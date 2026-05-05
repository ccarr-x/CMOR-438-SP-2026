import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.processing.postprocessing import PostProcessor

@pytest.fixture

def sample_data():
    """Simple dataset for postprocessing testing."""
    predictions = np.array([0.2, 0.5, 0.8, 0.1, 0.9])
    return predictions

def test_postprocessor_threshold(sample_data):
    predictions = sample_data
    postprocessor = PostProcessor(threshold=0.5)
    binary_predictions = postprocessor.threshold(predictions)
    expected = np.array([0, 0, 1, 0, 1])
    assert np.array_equal(binary_predictions, expected), f"Expected {expected}, but got {binary_predictions}"

def test_postprocessor_threshold_edge_case(sample_data):
    predictions = sample_data
    postprocessor = PostProcessor(threshold=0.5)
    binary_predictions = postprocessor.threshold(predictions)
    expected = np.array([0, 0, 1, 0, 1])
    assert np.array_equal(binary_predictions, expected), f"Expected {expected}, but got {binary_predictions}"

def test_postprocessor_threshold_invalid_input(sample_data):
    predictions = sample_data
    postprocessor = PostProcessor(threshold=0.5)
    with pytest.raises(ValueError):
        postprocessor.threshold([0.2, 0.5, "invalid", 0.1, 0.9])  # Non-numeric value in predictions
    with pytest.raises(ValueError):
        postprocessor.threshold([0.2, 0.5, -0.1, 0.1, 0.9])  # Negative value in predictions
    with pytest.raises(ValueError):
        postprocessor.threshold([0.2, 0.5, 1.1, 0.1, 0.9])  # Value greater than 1 in predictions

def test_postprocessor_threshold_all_below(sample_data):
    predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    postprocessor = PostProcessor(threshold=0.6)
    binary_predictions = postprocessor.threshold(predictions)
    expected = np.array([0, 0, 0, 0, 0])
    assert np.array_equal(binary_predictions, expected), f"Expected {expected}, but got {binary_predictions}"

def test_postprocessor_threshold_all_above(sample_data):
    predictions = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    postprocessor = PostProcessor(threshold=0.5)
    binary_predictions = postprocessor.threshold(predictions)
    expected = np.array([1, 1, 1, 1, 1])
    assert np.array_equal(binary_predictions, expected), f"Expected {expected}, but got {binary_predictions}"