"""
src/utils/cascade_utils.py
===========================
Haar cascade downloader with correct URLs.

v2 Bug Fix:
    haarcascade_mcs_mouth.xml does NOT exist in the OpenCV `master` branch.
    It is only available in the `2.4.13.7` branch.
    The three eye/face cascades come from `master`.

    Original broken URL:
        https://raw.githubusercontent.com/opencv/opencv/2.4/data/...
        (2.4 is not a valid branch tag — returns 404)

    Fixed URLs:
        master branch → face + eye cascades
        2.4.13.7 tag  → mouth cascade only
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

# Correct URLs (bug-fixed)
_MASTER = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
_V24    = "https://raw.githubusercontent.com/opencv/opencv/2.4.13.7/data/haarcascades/"

CASCADE_URLS: dict[str, str] = {
    "haarcascade_frontalface_alt.xml":  _MASTER,
    "haarcascade_lefteye_2splits.xml":  _MASTER,
    "haarcascade_righteye_2splits.xml": _MASTER,
    "haarcascade_mcs_mouth.xml":        _V24,   # FIX: only in 2.4 branch
}


def download_cascades(local_dir: str | Path = "cascades") -> dict[str, Path]:
    """
    Download all required Haar cascade XML files.

    Args:
        local_dir: Directory to save cascades. Created if absent.

    Returns:
        Dict mapping filename → local Path.
    """
    save_dir = Path(local_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    for fname, base_url in CASCADE_URLS.items():
        save_path = save_dir / fname
        if save_path.exists():
            logger.info("  ✅ Already exists: %s", fname)
        else:
            url = base_url + fname
            logger.info("  Downloading %s …", fname)
            try:
                urllib.request.urlretrieve(url, str(save_path))
            except Exception as e:
                logger.error("  ❌ Failed to download %s: %s", fname, e)
                continue

        # Validate cascade
        cc = cv2.CascadeClassifier(str(save_path))
        if cc.empty():
            logger.error("  ❌ Invalid cascade XML: %s", fname)
        else:
            logger.info("  ✅ Valid: %s", fname)
            paths[fname] = save_path

    return paths


def load_cascades(local_dir: str | Path = "cascades") -> dict[str, cv2.CascadeClassifier]:
    """
    Load all cascades from disk, downloading any that are missing.

    Returns:
        Dict mapping role → CascadeClassifier:
            'face', 'left_eye', 'right_eye', 'mouth'
    """
    paths = download_cascades(local_dir)

    def _load(fname: str) -> cv2.CascadeClassifier:
        p = Path(local_dir) / fname
        cc = cv2.CascadeClassifier(str(p))
        if cc.empty():
            raise RuntimeError(f"Failed to load cascade: {p}")
        return cc

    return {
        "face":      _load("haarcascade_frontalface_alt.xml"),
        "left_eye":  _load("haarcascade_lefteye_2splits.xml"),
        "right_eye": _load("haarcascade_righteye_2splits.xml"),
        "mouth":     _load("haarcascade_mcs_mouth.xml"),
    }
