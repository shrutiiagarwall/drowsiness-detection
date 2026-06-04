"""
src/preprocessing/loader.py
============================
Dataset loading for both eye-state and yawn datasets.
Includes CLAHE enhancement and square-padding for mouth ROIs.

Public code — no proprietary logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLAHE Enhancement
# ---------------------------------------------------------------------------

def apply_clahe(
    gray_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).
    Normalises local contrast to improve eye-open/closed discrimination
    under varied illumination conditions.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)


# ---------------------------------------------------------------------------
# Eye Dataset Loader
# ---------------------------------------------------------------------------

def load_dataset(
    root: Path,
    categories: list[str],
    img_size: int = 48,
    apply_enhancement: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load eye-state images (Closed / Open).

    Expected layout::
        root/
          Closed/  *.jpg | *.png
          Open/

    Returns:
        X: uint8 (N, H, W)
        Y: int32 (N,)
    """
    X: list[np.ndarray] = []
    Y: list[int] = []
    corrupt = 0

    for label_idx, category in enumerate(categories):
        folder = root / category
        if not folder.exists():
            logger.warning("Folder not found: %s", folder)
            continue
        files = sorted(folder.glob("*.*"))
        logger.info("  Loading %s: %d files…", category, len(files))

        for img_path in tqdm(files, desc=category, leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if apply_enhancement:
                gray = apply_clahe(gray)
            gray = cv2.resize(gray, (img_size, img_size),
                              interpolation=cv2.INTER_AREA)
            X.append(gray)
            Y.append(label_idx)

    logger.info("Loaded %d images | Corrupt: %d", len(X), corrupt)
    return np.array(X, dtype=np.uint8), np.array(Y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Yawn Dataset Loader  (square-pad for wide mouth ROIs)
# ---------------------------------------------------------------------------

def load_yawn_dataset(
    root: Path,
    categories: list[str],
    img_size: int = 64,
    apply_enhancement: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load yawn/no_yawn images with square-padding.

    Mouth images are wider than tall; padding to square before resize
    prevents aspect-ratio distortion that degrades classifier accuracy.

    Expected layout::
        root/
          no_yawn/
          yawn/
    """
    X: list[np.ndarray] = []
    Y: list[int] = []
    corrupt = 0

    for label_idx, category in enumerate(categories):
        folder = root / category
        if not folder.exists():
            logger.warning("Yawn folder not found: %s", folder)
            continue
        files = sorted(folder.glob("*.*"))
        logger.info("  Loading %s: %d files…", category, len(files))

        for img_path in tqdm(files, desc=category, leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Square-pad (mouth is wider than tall)
            h, w = gray.shape
            side = max(h, w)
            padded = np.zeros((side, side), dtype=np.uint8)
            y_off = (side - h) // 2
            x_off = (side - w) // 2
            padded[y_off : y_off + h, x_off : x_off + w] = gray

            if apply_enhancement:
                padded = apply_clahe(padded)

            padded = cv2.resize(padded, (img_size, img_size),
                                interpolation=cv2.INTER_AREA)
            X.append(padded)
            Y.append(label_idx)

    logger.info("Yawn dataset: %d images | Corrupt: %d", len(X), corrupt)
    return np.array(X, dtype=np.uint8), np.array(Y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Dataset Inventory
# ---------------------------------------------------------------------------

def dataset_inventory(root: Path, categories: list[str]) -> dict:
    """Return per-class image counts."""
    stats: dict = {}
    for cat in categories:
        folder = root / cat
        if not folder.exists():
            stats[cat] = {"count": 0}
            continue
        files = list(folder.glob("*.*"))
        stats[cat] = {"count": len(files), "extensions": {f.suffix for f in files}}
    return stats
