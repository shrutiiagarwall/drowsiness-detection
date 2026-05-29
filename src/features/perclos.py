"""
src/features/perclos.py
========================
PERCLOS (Percentage of Eye Closure) sliding-window scorer
and multi-signal temporal fusion engine.

PERCLOS is a clinically validated drowsiness indicator defined as
the proportion of frames within a rolling time window in which the
eyes are classified as "closed".

Reference:
    Wierwille & Ellsworth (1994). "Evaluation of Driver Drowsiness
    by Trained Raters." Accident Analysis & Prevention, 26(5), 571-581.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class FusionResult:
    """Output of one FusionDrowsinessScorer.update() call."""
    perclos: float          # % eye closure in rolling window  ∈ [0, 1]
    yawn_rate: float        # % yawning frames in rolling window ∈ [0, 1]
    yawn_count: int         # cumulative yawn events this session
    fusion_score: float     # weighted combination ∈ [0, 1]
    status: str             # "OK" | "WARNING" | "ALERT"


class FusionDrowsinessScorer:
    """
    Temporal fusion of eye-state and yawn signals.

    Fusion formula:
        score = eye_weight × PERCLOS + yawn_weight × yawn_rate

    Default weights (eye=0.6, yawn=0.4) reflect the clinical literature
    which gives primary weight to the PERCLOS metric.

    Args:
        window_frames:    Number of frames in the sliding window (default 90,
                          ≈ 3 s at 30 fps).
        eye_weight:       Weight assigned to PERCLOS signal.
        yawn_weight:      Weight assigned to yawn-rate signal.
        alert_threshold:  Fusion score above which status is ALERT.
        warning_threshold: Fusion score above which status is WARNING.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, eyes_closed: bool, yawning: bool) -> FusionResult:
        """
        Ingest one frame's binary detections and return updated scores.

        Args:
            eyes_closed: True if the eye model predicts "Closed".
            yawning:     True if the yawn model predicts "yawn".

        Returns:
            FusionResult with all current scores and alert status.
        """
        self._eye_buf.append(int(eyes_closed))
        self._yawn_buf.append(int(yawning))

        # Rising edge → count a new yawn event
        if yawning and not self._prev_yawning:
            self._yawn_count += 1
        self._prev_yawning = yawning

        perclos = self._compute_perclos()
        yawn_rate = self._compute_yawn_rate()
        fusion_score = (
            self._eye_weight * perclos + self._yawn_weight * yawn_rate
        )

        if fusion_score >= self._alert_threshold:
            status = "ALERT"
        elif fusion_score >= self._warning_threshold:
            status = "WARNING"
        else:
            status = "OK"

        return FusionResult(
            perclos=perclos,
            yawn_rate=yawn_rate,
            yawn_count=self._yawn_count,
            fusion_score=fusion_score,
            status=status,
        )

    def reset(self) -> None:
        """Clear all buffers (call between driving sessions)."""
        self._eye_buf.clear()
        self._yawn_buf.clear()
        self._yawn_count = 0
        self._prev_yawning = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_perclos(self) -> float:
        if not self._eye_buf:
            return 0.0
        return sum(self._eye_buf) / len(self._eye_buf)

    def _compute_yawn_rate(self) -> float:
        if not self._yawn_buf:
            return 0.0
        return sum(self._yawn_buf) / len(self._yawn_buf)
