"""
src/utils/drift_detector.py
=============================
ADWIN (ADaptive WINdowing) concept drift detector wrapper.

ADWIN monitors a binary data stream (correct/incorrect predictions)
and raises an alert when the error rate has statistically changed,
signalling that the model may need retraining on new data.

Reference:
    Bifet & Gavalda (2007). "Learning from Time-Changing Data with
    Adaptive Windowing." SDM 2007.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DriftStatus:
    drift_detected: bool
    warning_detected: bool
    current_error_rate: float
    n_samples_seen: int
    n_drift_events: int


class ADWINDriftDetector:
    """
    Lightweight ADWIN wrapper using the `river` library if available,
    with a pure-Python fallback for environments where `river` is absent.

    Usage::

        detector = ADWINDriftDetector()
        for y_true, y_pred in stream:
            status = detector.update(int(y_true != y_pred))  # 1=error, 0=correct
            if status.drift_detected:
                print("Drift detected — consider retraining.")
    """

    def __init__(self, delta: float = 0.002) -> None:
        self._delta = delta
        self._n_samples = 0
        self._n_drift_events = 0
        self._errors: list[int] = []
        self._adwin = self._try_load_river_adwin(delta)

    # ------------------------------------------------------------------

    def update(self, error_bit: int) -> DriftStatus:
        """
        Ingest one observation (1 = prediction error, 0 = correct).

        Returns:
            DriftStatus with current error rate and drift flags.
        """
        self._n_samples += 1
        self._errors.append(error_bit)

        drift_detected = False
        warning_detected = False

        if self._adwin is not None:
            self._adwin.update(error_bit)
            drift_detected = self._adwin.drift_detected
        else:
            drift_detected = self._fallback_drift_check()

        if drift_detected:
            self._n_drift_events += 1
            logger.warning(
                "⚠️  Concept drift detected at sample %d "
                "(total drift events: %d).",
                self._n_samples, self._n_drift_events,
            )

        current_error_rate = (
            sum(self._errors[-100:]) / min(len(self._errors), 100)
        )

        return DriftStatus(
            drift_detected=drift_detected,
            warning_detected=warning_detected,
            current_error_rate=current_error_rate,
            n_samples_seen=self._n_samples,
            n_drift_events=self._n_drift_events,
        )

    def reset(self) -> None:
        self._errors.clear()
        self._n_samples = 0
        if self._adwin is not None:
            self._adwin = self._try_load_river_adwin(self._delta)

    # ------------------------------------------------------------------

    @staticmethod
    def _try_load_river_adwin(delta: float):
        try:
            from river.drift import ADWIN
            return ADWIN(delta=delta)
        except ImportError:
            logger.info(
                "river library not installed — using fallback drift detector. "
                "Install with: pip install river"
            )
            return None

    def _fallback_drift_check(self) -> bool:
        """
        Simplified fallback: detect if recent error rate exceeds a threshold.
        Not as statistically rigorous as ADWIN but functional without `river`.
        """
        window = 50
        if len(self._errors) < window * 2:
            return False
        recent = sum(self._errors[-window:]) / window
        baseline = sum(self._errors[-window * 2 : -window]) / window
        return (recent - baseline) > 0.15  # 15pp increase → flag drift
