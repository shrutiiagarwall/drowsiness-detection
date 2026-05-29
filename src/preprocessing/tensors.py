"""
src/preprocessing/tensors.py
==============================
Convert numpy arrays to normalised TensorFlow tensors and tf.data pipelines.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def prepare_tensors(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise uint8 images to float32 [0, 1] and add channel dimension.

    Args:
        X: uint8 array (N, H, W).
        Y: int label array (N,).

    Returns:
        X_f: float32 (N, H, W, 1).
        Y_i: int32 (N,).
    """
    X_f = X.astype(np.float32) / 255.0
    X_f = X_f[..., np.newaxis]
    return X_f, Y.astype(np.int32)


def make_tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a prefetched tf.data.Dataset from numpy arrays."""
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
