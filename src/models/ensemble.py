"""
src/models/ensemble.py
=======================
Soft-voting ensemble that combines predictions from multiple
Keras models (CBAM-CNN, MobileNetV2, EfficientNetB0).
"""

from __future__ import annotations

import numpy as np
from tensorflow.keras import Model


class SoftVotingEnsemble:
    """
    Soft-voting ensemble over a list of trained Keras classifiers.

    Each model contributes equally (uniform weights). Extends trivially
    to non-uniform weights via the ``weights`` argument.

    Args:
        models:  List of compiled, loaded Keras models.
        weights: Optional per-model weights (must sum to 1). If None,
                 uniform weighting is used.
    """

    def __init__(
        self,
        models: list[Model],
        weights: list[float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("len(weights) must equal len(models).")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.")
        self._models = models
        self._weights = weights or [1.0 / len(models)] * len(models)

    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return averaged class probabilities.

        Args:
            X: Input array shape (N, H, W, 1), float32, values ∈ [0, 1].

        Returns:
            Probability array shape (N, num_classes).
        """
        proba_sum = np.zeros(
            (X.shape[0], self._models[0].output_shape[-1]), dtype=np.float64
        )
        for model, w in zip(self._models, self._weights):
            proba_sum += w * model.predict(X, verbose=0)
        return proba_sum.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions (argmax of averaged probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def __repr__(self) -> str:
        model_names = [m.name for m in self._models]
        return (
            f"SoftVotingEnsemble(models={model_names}, "
            f"weights={self._weights})"
        )
