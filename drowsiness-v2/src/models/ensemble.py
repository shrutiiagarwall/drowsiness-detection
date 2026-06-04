"""
src/models/ensemble.py
=======================
Soft-voting ensemble over multiple Keras classifiers.
"""

from __future__ import annotations

import numpy as np
from tensorflow.keras import Model


class SoftVotingEnsemble:
    def __init__(self, models: list[Model], weights: list[float] | None = None):
        if not models:
            raise ValueError("At least one model required.")
        self._models = models
        self._weights = weights or [1.0 / len(models)] * len(models)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_classes = self._models[0].output_shape[-1]
        proba_sum = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for model, w in zip(self._models, self._weights):
            proba_sum += w * model.predict(X, verbose=0)
        return proba_sum.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
