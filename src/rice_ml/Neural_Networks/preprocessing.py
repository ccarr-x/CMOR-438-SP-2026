"""TensorFlow/Keras preprocessing helpers for neural network workflows."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
#import tensorflow as tf

ArrayLike = Union[np.ndarray, list]


def _get_tensorflow():
    """Import TensorFlow so rice_ml can still load"""
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is suggested for `flatten_images` and `one_hot_encode_labels`."
        ) from exc
    return tf


def flatten_images(images: ArrayLike) -> np.ndarray:
    """
    Flatten image batches to 2D arrays suitable for dense networks.

    Parameters
    ----------
    images : array-like
        Image batch with shape (n_samples, ...), e.g. (n_samples, height, width[, channels]).

    Returns
    -------
    np.ndarray
        Flattened image batch with shape (n_samples, n_features).
    """
    tf = _get_tensorflow()
    image_tensor = tf.convert_to_tensor(images)

    if image_tensor.shape.rank is None or image_tensor.shape.rank < 2:
        raise ValueError("images must include a sample dimension and at least one feature dimension.")

    flattened = tf.keras.layers.Flatten()(image_tensor)
    return flattened.numpy()


def one_hot_encode_labels(labels: ArrayLike, num_classes: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode integer class labels using tf.keras.utils.to_categorical.

    Parameters
    ----------
    labels : array-like
        Integer class labels with shape (n_samples,).
    num_classes : int or None, default=None
        Total number of classes. If None, inferred by Keras from labels.

    Returns
    -------
    np.ndarray
        One-hot encoded labels with shape (n_samples, n_classes).
    """
    tf = _get_tensorflow()
    label_tensor = tf.convert_to_tensor(labels)
    label_array = np.asarray(label_tensor.numpy(), dtype=int).reshape(-1)

    if label_array.size == 0:
        raise ValueError("labels cannot be empty.")

    if np.any(label_array < 0):
        raise ValueError("labels must be non-negative integers.")

    encoded = tf.keras.utils.to_categorical(label_array, num_classes=num_classes)
    return np.asarray(encoded, dtype=np.float32)
