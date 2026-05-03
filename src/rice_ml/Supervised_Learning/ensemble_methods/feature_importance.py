"""Feature importance utilities for ensemble methods."""

from typing import List, Optional, Union
import numpy as np

from .bagging_forest_class import BaggingForestClassifier
from .random_forest_class import RandomForestClassifier, RandomForestRegressor
from .hard_voting_class import HardVotingClassifier

def get_feature_importance_bagging_forest(ensemble: BaggingForestClassifier) -> Optional[np.ndarray]:
    """
    Compute feature importances for BaggingForestClassifier.

    Parameters
    ----------
    ensemble : BaggingForestClassifier
        The fitted bagging forest classifier.

    Returns
    -------
    feature_importances : ndarray of shape (n_features,)
        The feature importances. Returns None if trees don't have feature_importances_.
    """
    if not hasattr(ensemble.trees[0], 'feature_importances_'):
        return None
    importances = np.array([tree.feature_importances_ for tree in ensemble.trees])
    return np.mean(importances, axis=0)

def get_feature_importance_random_forest(ensemble: Union[RandomForestClassifier, RandomForestRegressor]) -> Optional[np.ndarray]:
    """
    Compute feature importances for RandomForestClassifier or RandomForestRegressor.

    Parameters
    ----------
    ensemble : RandomForestClassifier or RandomForestRegressor
        The fitted random forest.

    Returns
    -------
    feature_importances : ndarray of shape (n_features,)
        The feature importances. Returns None if trees don't have feature_importances_.
    """
    if not hasattr(ensemble.trees[0], 'feature_importances_'):
        return None
    importances = np.array([tree.feature_importances_ for tree in ensemble.trees])
    return np.mean(importances, axis=0)

def get_feature_importance_hard_voting(ensemble: HardVotingClassifier) -> Optional[np.ndarray]:
    """
    Compute feature importances for HardVotingClassifier by averaging importances from base estimators.

    Parameters
    ----------
    ensemble : HardVotingClassifier
        The fitted hard voting classifier.

    Returns
    -------
    feature_importances : ndarray of shape (n_features,)
        The averaged feature importances. Returns None if estimators don't have feature_importances_.
    """
    importances_list = []
    for estimator in ensemble.estimators:
        if hasattr(estimator, 'feature_importances_'):
            importances_list.append(estimator.feature_importances_)
        else:
            return None  # If any estimator doesn't have it, return None
    if not importances_list:
        return None
    importances = np.array(importances_list)
    return np.mean(importances, axis=0)

# Convenience functions that work with any ensemble
def get_feature_importance(ensemble) -> Optional[np.ndarray]:
    """
    Compute feature importances for any supported ensemble method.

    Parameters
    ----------
    ensemble : ensemble object
        The fitted ensemble (BaggingForestClassifier, RandomForestClassifier, etc.)

    Returns
    -------
    feature_importances : ndarray of shape (n_features,) or None
        The feature importances if available, else None.
    """
    if isinstance(ensemble, BaggingForestClassifier):
        return get_feature_importance_bagging_forest(ensemble)
    elif isinstance(ensemble, (RandomForestClassifier, RandomForestRegressor)):
        return get_feature_importance_random_forest(ensemble)
    elif isinstance(ensemble, HardVotingClassifier):
        return get_feature_importance_hard_voting(ensemble)
    else:
        raise ValueError(f"Unsupported ensemble type: {type(ensemble)}")