import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rice_ml.Supervised_Learning.ensemble_methods.random_forest_class import RandomForestClassifier, RandomForestRegressor 

@pytest.fixture
def sample_data_classification():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate
    return X, y

def test_random_forest_classification(sample_data_classification):
    X, y = sample_data_classification
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_random_forest_classification_unseen_data(sample_data_classification):
    X, y = sample_data_classification
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_random_forest_classification_invalid_input():
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels

def test_random_forest_classification_predict_before_fit():
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])

def test_random_forest_classification_convergence(sample_data_classification):
    X, y = sample_data_classification
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_random_forest_classification_weights(sample_data_classification):
    X, y = sample_data_classification
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    assert len(model.estimators_) == 10, f"Expected 10 estimators, but got {len(model.estimators_)}"
    for estimator in model.estimators_:
        assert hasattr(estimator, "predict"), f"Each estimator should have a predict method, but got {type(estimator)}"

# --- Regression tests ---
def test_random_forest_regression():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])  # Perfect linear relationship
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y), f"Expected {y}, but got {predictions}"

def test_random_forest_regression_unseen_data(): 
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])  # Perfect linear relationship
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    unseen_X = np.array([[0.5], [1.5], [2.5]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(predictions >= 0) and np.all(predictions <= 3), f"Expected predictions to be between 0 and 3, but got {predictions}"

def test_random_forest_regression_invalid_input():
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0, 1, 2])  # Mismatched number of samples and labels

def test_random_forest_regression_predict_before_fit():
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5], [1.5], [2.5]])

def test_random_forest_regression_convergence():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])  # Perfect linear relationship
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.allclose(predictions, y), f"Expected {y}, but got {predictions}"

def test_random_forest_regression_weights():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])  # Perfect linear relationship
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    assert len(model.estimators_) == 10, f"Expected 10 estimators, but got {len(model.estimators_)}"
    for estimator in model.estimators_:
        assert hasattr(estimator, "predict"), f"Each estimator should have a predict method, but got {type(estimator)}"