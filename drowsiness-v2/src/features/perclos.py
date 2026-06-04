"""
src/features/perclos.py
========================
PERCLOS scorer and multi-signal temporal fusion engine.

v2 Bug Fix:
    PERCLOS = eyes closed > 70% of window.
    Original code used `l_state == 0 OR r_state == 0` meaning a SINGLE
    closed eye triggered drowsiness. Clinically, PERCLOS requires BOTH
    eyes closed. Fixed to `l_state == 0 AND r_state == 0`.

    This is passed in as the `eyes_closed` boolean from the caller —
    the caller must evaluate `left_closed AND right_closed` before
    calling update().

score_history is now maintained for the Gradio timeline plot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class FusionResult:
    perclos: float
    yawn_rate: float
    yawn_count: int
    fusion_score: float
    status: str


class FusionDrowsinessScorer:
    """
    Temporal fusion of eye-state and yawn signals.

    Fusion formula:
        score = eye_weight × PERCLOS + yawn_weight × yawn_rate

    The caller is responsible for computing `eyes_closed` correctly:

        # CORRECT (v2 fix):
        eyes_closed = (left_state == 0) and (right_state == 0)

        # WRONG (original bug):
        eyes_closed = (left_state == 0) or (right_state == 0)

    Args:
        window_frames:     Sliding window size (default 90 ≈ 3 s @ 30 fps).
        eye_weight:        Weight for PERCLOS signal (default 0.6).
        yawn_weight:       Weight for yawn-rate signal (default 0.4).
        alert_threshold:   Score threshold for ALERT status.
        warning_threshold: Score threshold for WARNING status.
    """

    def __init__(
        self,
        window_frames: int = 90,
        eye_weight: float = 0.6,
        yawn_weight: float = 0.4,
        alert_threshold: float = 0.65,
        warning_threshold: float = 0.45,
    ) -> None:
        self._eye_buf: deque[int] = deque(maxlen=window_frames)
        self._yawn_buf: deque[int] = deque(maxlen=window_frames)
        self._eye_weight = eye_weight
        self._yawn_weight = yawn_weight
        self._alert_threshold = alert_threshold
        self._warning_threshold = warning_threshold
        self._yawn_count: int = 0
        self._prev_yawning: bool = False
        # Maintained for Gradio timeline plot
        self.score_history: list[float] = []

    def update(self, eyes_closed: bool, yawning: bool) -> dict:
        """
        Ingest one frame and return updated scores as a dict.

        Returns dict for backward compatibility with notebook code that
        accesses result['fusion_score'], result['perclos'], etc.

        Args:
            eyes_closed: True only if BOTH eyes are classified as closed.
            yawning:     True if yawn model predicts yawn.
        """
        self._eye_buf.append(int(eyes_closed))
        self._yawn_buf.append(int(yawning))

        # Rising edge = new yawn event
        if yawning and not self._prev_yawning:
            self._yawn_count += 1
        self._prev_yawning = yawning

        perclos = self._perclos()
        yawn_rate = self._yawn_rate()
        fusion_score = self._eye_weight * perclos + self._yawn_weight * yawn_rate

        if fusion_score >= self._alert_threshold:
            status = "ALERT"
        elif fusion_score >= self._warning_threshold:
            status = "WARNING"
        else:
            status = "OK"

        self.score_history.append(fusion_score)

        return {
            "perclos": perclos,
            "yawn_rate": yawn_rate,
            "yawn_count": self._yawn_count,
            "fusion_score": fusion_score,
            "status": status,
        }

    def reset(self) -> None:
        """Clear buffers between driving sessions."""
        self._eye_buf.clear()
        self._yawn_buf.clear()
        self._yawn_count = 0
        self._prev_yawning = False
        self.score_history.clear()

    def _perclos(self) -> float:
        if not self._eye_buf:
            return 0.0
        return sum(self._eye_buf) / len(self._eye_buf)

    def _yawn_rate(self) -> float:
        if not self._yawn_buf:
            return 0.0
        return sum(self._yawn_buf) / len(self._yawn_buf)
