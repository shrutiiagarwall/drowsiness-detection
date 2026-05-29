"""
app/inference_engine.py
========================
Real-time inference engine used by both the Gradio UI and the
FastAPI server. Handles model loading, fallback to mock mode when
weights are absent, and per-frame processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Loader (graceful mock fallback)
# ---------------------------------------------------------------------------

def load_model_or_mock(weights_path: str | Path):
    """
    Attempt to load the CBAM-CNN from disk.
    Returns a real Keras model if weights exist, otherwise a MockModel.

    This lets the entire app run in demo mode without weights present.
    """
    path = Path(weights_path)
    if path.exists():
        try:
            import tensorflow as tf
            from src.models.cbam_cnn import (
                ReduceMeanLayer,
                ReduceMaxLayer,
            )

            model = tf.keras.models.load_model(
                str(path),
                custom_objects={
                    "ReduceMeanLayer": ReduceMeanLayer,
                    "ReduceMaxLayer": ReduceMaxLayer,
                },
            )
            logger.info("✅ Model loaded from %s", path)
            return model
        except Exception as exc:
            logger.warning("Could not load model: %s — using mock.", exc)

    logger.warning(
        "⚠️  No model weights found at '%s'. Running in DEMO (mock) mode.", path
    )
    return MockModel()


class MockModel:
    """
    Deterministic mock that returns plausible-looking predictions.
    Used when real weights are not present (demo / CI mode).
    """

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        n = X.shape[0]
        # Derive pseudo-label from mean pixel intensity:
        # dark patch → likely closed; bright → likely open
        means = X.reshape(n, -1).mean(axis=1)
        closed_prob = np.clip(1.0 - means * 1.8, 0.05, 0.95)
        open_prob = 1.0 - closed_prob
        return np.stack([closed_prob, open_prob], axis=1).astype(np.float32)

    @property
    def name(self) -> str:
        return "MockModel"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(
    img_bgr: np.ndarray,
    img_size: int = 48,
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Convert a BGR frame to a normalised single-channel tensor.

    Returns:
        float32 array shape (1, img_size, img_size, 1), values ∈ [0, 1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if apply_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe_obj.apply(gray)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = gray.astype(np.float32) / 255.0
    return tensor.reshape(1, img_size, img_size, 1)


def preprocess_pil(pil_image, img_size: int = 48, apply_clahe: bool = True):
    """PIL Image → model-ready tensor (used by Gradio)."""
    import numpy as np
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr = img_array[:, :, ::-1]  # RGB → BGR
    return preprocess_frame(img_bgr, img_size=img_size, apply_clahe=apply_clahe)


# ---------------------------------------------------------------------------
# Single-frame inference result
# ---------------------------------------------------------------------------

class InferenceResult:
    def __init__(
        self,
        label: int,
        label_name: str,
        confidence: float,
        is_drowsy: bool,
        is_mock: bool = False,
    ) -> None:
        self.label = label
        self.label_name = label_name
        self.confidence = confidence
        self.is_drowsy = is_drowsy
        self.is_mock = is_mock

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_name": self.label_name,
            "confidence": round(self.confidence, 4),
            "is_drowsy": self.is_drowsy,
            "demo_mode": self.is_mock,
        }

    def __repr__(self) -> str:
        mode = " [DEMO]" if self.is_mock else ""
        return (
            f"InferenceResult({self.label_name}, "
            f"conf={self.confidence:.3f}){mode}"
        )


def run_inference(model, tensor: np.ndarray) -> InferenceResult:
    """Run model on a preprocessed tensor and return a structured result."""
    is_mock = isinstance(model, MockModel)
    probs = model.predict(tensor, verbose=0)[0]
    label = int(np.argmax(probs))
    categories = ["Closed", "Open"]
    return InferenceResult(
        label=label,
        label_name=categories[label],
        confidence=float(probs[label]),
        is_drowsy=(label == 0),
        is_mock=is_mock,
    )
