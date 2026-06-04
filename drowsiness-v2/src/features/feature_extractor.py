"""
src/features/feature_extractor.py
====================================
Public stub for the novel ocular feature engineering pipeline.

⚠️  INTELLECTUAL PROPERTY NOTICE
----------------------------------
This module exposes the public interface only. The proprietary
mathematical formulations, calibration thresholds, and optimised
kernel parameters are not included in this repository.

Full implementation available under research collaboration agreement.
Contact: [your-email@institution.edu]
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class OcularFeatureExtractor:
    """
    Stub implementation of the novel ocular feature extractor.

    What the real implementation does (described in paper Section 3.2):
        1. Multi-scale frequency analysis of the eye-region patch.
        2. Landmark-conditioned texture sampling.
        3. Proprietary temporal smoothing transform.
        4. Attention-weighted saliency descriptor for CNN augmentation.

    This stub returns semantically consistent placeholder outputs
    so the pipeline runs end-to-end for demonstration purposes.

    To swap in the real implementation (interface-identical):
        from src.features._private_extractor import _RealOcularFeatureExtractor
        extractor = _RealOcularFeatureExtractor(config)
    """

    DESCRIPTOR_DIM: int = 128

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._frame_buffer: list[np.ndarray] = []
        self._window_size: int = self._config.get("window_size", 10)
        logger.info("OcularFeatureExtractor initialised (public stub — proprietary logic not included).")

    def extract(self, eye_patch: np.ndarray) -> dict:
        """
        Extract feature vector from a single preprocessed eye patch.

        Args:
            eye_patch: float32 array (H, W) or (H, W, 1), values in [0,1].

        Returns:
            dict with keys: descriptor (float32 array), temporal_score (float),
                            saliency_map (float32 array).
        """
        patch = self._normalise(eye_patch)
        self._frame_buffer.append(patch)
        if len(self._frame_buffer) > self._window_size:
            self._frame_buffer.pop(0)

        # STUB logic only — not research code
        descriptor = self._stub_descriptor(patch)
        temporal_score = self._stub_temporal_score()
        saliency_map = self._stub_saliency(patch)

        return {
            "descriptor": descriptor,
            "temporal_score": temporal_score,
            "saliency_map": saliency_map,
        }

    def reset(self) -> None:
        self._frame_buffer.clear()

    @staticmethod
    def _normalise(patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 3:
            patch = patch[..., 0]
        return patch.astype(np.float32)

    def _stub_descriptor(self, patch: np.ndarray) -> np.ndarray:
        flat = patch.ravel()
        stats = np.array([
            flat.mean(), flat.std(),
            np.percentile(flat, 25), np.percentile(flat, 75),
            float(np.sum(flat < 0.3)) / flat.size,
        ], dtype=np.float32)
        desc = np.zeros(self.DESCRIPTOR_DIM, dtype=np.float32)
        desc[: len(stats)] = stats
        return desc

    def _stub_temporal_score(self) -> float:
        if len(self._frame_buffer) < 2:
            return 0.5
        means = [f.mean() for f in self._frame_buffer]
        return float(np.clip(1.0 - np.var(means) * 10, 0.0, 1.0))

    def _stub_saliency(self, patch: np.ndarray) -> np.ndarray:
        import cv2
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        max_v = mag.max()
        return mag / max_v if max_v > 0 else mag
