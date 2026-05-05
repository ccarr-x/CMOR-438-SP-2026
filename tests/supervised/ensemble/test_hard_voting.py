import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from rice_ml.Supervised_Learning.ensemble_methods.hard_voting_class import HardVotingClassifier
from rice_ml.Supervised_Learning.decision_tree_class import DecisionTree


def make_hard_voting_model():
    estimators = [
        ("dt1", DecisionTree(max_depth=1)),
        ("dt2", DecisionTree(max_depth=2)),
        ("dt3", DecisionTree(max_depth=3)),
    ]
    return HardVotingClassifier(estimators=estimators)


@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND logic gate
    return X, y

def test_hard_voting(sample_data):
    X, y = sample_data
    model = make_hard_voting_model()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_hard_voting_unseen_data(sample_data):
    X, y = sample_data
    model = make_hard_voting_model()
    model.fit(X, y)
    unseen_X = np.array([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])
    predictions = model.predict(unseen_X)
    assert predictions.shape == (3,), f"Expected predictions shape (3,), but got {predictions.shape}"
    assert np.all(np.isin(predictions, [0, 1])), f"Expected predictions to be binary (0 or 1), but got {predictions}"

def test_hard_voting_invalid_input():
    model = make_hard_voting_model()
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0])  # Mismatched number of samples and labels
    with pytest.raises(ValueError):
        model.fit([[0, 0], [0, 1]], [0, 1, 1])  # Mismatched number of samples and labels

def test_hard_voting_predict_before_fit():
    model = make_hard_voting_model()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict([[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]])

def test_hard_voting_convergence(sample_data):
    X, y = sample_data
    model = make_hard_voting_model()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y), f"Expected {y}, but got {predictions}"

def test_hard_voting_weights(sample_data):
    X, y = sample_data
    model = make_hard_voting_model()
    model.fit(X, y)
    assert len(model.estimators_) == 3, f"Expected 3 estimators, but got {len(model.estimators_)}"
    for name, estimator in model.estimators_:
        assert hasattr(estimator, "predict"), f"Each estimator should have a predict method, but got {type(estimator)}"