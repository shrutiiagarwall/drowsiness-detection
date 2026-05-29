"""
src/preprocessing/loader.py
============================
Dataset loading with CLAHE contrast enhancement.

Public interface — no proprietary logic here.
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

    Normalises local contrast to improve eye-open / closed discrimination
    under varied illumination conditions — a common failure mode of naive
    grayscale thresholding approaches.

    Args:
        gray_img:   Single-channel uint8 grayscale image.
        clip_limit: CLAHE clip limit (higher → more contrast).
        tile_grid:  Grid size for local histogram computation.

    Returns:
        Enhanced grayscale image (same dtype / shape as input).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------

def load_dataset(
    root: Path,
    categories: list[str],
    img_size: int = 48,
    apply_enhancement: bool = True,
    square_pad: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from a class-labelled folder structure.

    Expected layout::

        root/
          <categories[0]>/  *.jpg | *.png
          <categories[1]>/  ...

    Args:
        root:               Root directory containing class sub-folders.
        categories:         Ordered list of class names (index → label int).
        img_size:           Target square resolution after resize.
        apply_enhancement:  Apply CLAHE before resize.
        square_pad:         Pad to square before resize (useful for mouth ROIs).

    Returns:
        X: uint8 array of shape (N, img_size, img_size).
        Y: int32 label array of shape (N,).
    """
    X: list[np.ndarray] = []
    Y: list[int] = []
    corrupt = 0

    for label_idx, category in enumerate(categories):
        folder = root / category
        if not folder.exists():
            logger.warning("Category folder not found: %s", folder)
            continue

        files = sorted(folder.glob("*.*"))
        logger.info("  Loading %s: %d files…", category, len(files))

        for img_path in tqdm(files, desc=category, leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if square_pad:
                h, w = gray.shape
                side = max(h, w)
                padded = np.zeros((side, side), dtype=np.uint8)
                y_off = (side - h) // 2
                x_off = (side - w) // 2
                padded[y_off : y_off + h, x_off : x_off + w] = gray
                gray = padded

            if apply_enhancement:
                gray = apply_clahe(gray)

            gray = cv2.resize(
                gray, (img_size, img_size), interpolation=cv2.INTER_AREA
            )
            X.append(gray)
            Y.append(label_idx)

    logger.info(
        "Loaded %d images | Corrupt/skipped: %d", len(X), corrupt
    )
    return np.array(X, dtype=np.uint8), np.array(Y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Dataset Inventory
# ---------------------------------------------------------------------------

def dataset_inventory(root: Path, categories: list[str]) -> dict:
    """Return per-class image counts and basic statistics."""
    stats: dict = {}
    for cat in categories:
        folder = root / cat
        if not folder.exists():
            stats[cat] = {"count": 0, "path": str(folder)}
            continue
        files = list(folder.glob("*.*"))
        exts = {f.suffix.lower() for f in files}
        stats[cat] = {
            "count": len(files),
            "path": str(folder),
            "extensions": exts,
        }
    return stats
