"""
app/inference_engine.py
========================
Model loading, preprocessing, and inference engine for both
eye-state and yawn classifiers.

Gracefully falls back to MockModel when weights are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock Models (demo mode)
# ---------------------------------------------------------------------------

class MockEyeModel:
    """Deterministic eye-state mock based on mean pixel intensity."""
    name = "MockEyeModel"

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        n = X.shape[0]
        means = X.reshape(n, -1).mean(axis=1)
        closed_prob = np.clip(1.0 - means * 1.8, 0.05, 0.95)
        return np.stack([closed_prob, 1.0 - closed_prob], axis=1).astype(np.float32)


class MockYawnModel:
    """Deterministic yawn mock (always predicts 'no yawn')."""
    name = "MockYawnModel"

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        n = X.shape[0]
        return np.tile([0.85, 0.15], (n, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_model_or_mock(weights_path: str | Path, mock_class=MockEyeModel):
    """Load Keras model from disk, or return a mock if absent."""
    path = Path(weights_path)
    if path.exists():
        try:
            import tensorflow as tf
            from src.models.cbam_cnn import ReduceMeanLayer, ReduceMaxLayer
            model = tf.keras.models.load_model(
                str(path),
                custom_objects={"ReduceMeanLayer": ReduceMeanLayer,
                                "ReduceMaxLayer": ReduceMaxLayer},
            )
            logger.info("✅ Model loaded: %s", path)
            return model
        except Exception as exc:
            logger.warning("Could not load model (%s) — using mock.", exc)
    else:
        logger.warning("⚠️  No weights at '%s' — running in DEMO mode.", path)
    return mock_class()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prep_eye(roi: np.ndarray, img_size: int = 48) -> np.ndarray:
    """BGR ROI → normalised eye tensor (1, H, W, 1)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0).reshape(1, img_size, img_size, 1)


def prep_mouth(roi: np.ndarray, img_size: int = 64) -> np.ndarray:
    """BGR mouth ROI → square-padded, normalised tensor (1, H, W, 1)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h, w = gray.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)
    padded[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    padded = clahe.apply(padded)
    padded = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return (padded.astype(np.float32) / 255.0).reshape(1, img_size, img_size, 1)


def preprocess_pil(pil_image, img_size: int = 48, apply_clahe: bool = True):
    """PIL Image → eye tensor for Gradio single-image mode."""
    import numpy as np
    img_bgr = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    return prep_eye(img_bgr, img_size=img_size)
