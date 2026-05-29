"""
src/preprocessing/augmentation.py
====================================
Albumentations-based augmentation pipeline and SMOTE oversampling.
"""

from __future__ import annotations

import logging

import albumentations as A
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Albumentations Pipeline
# ---------------------------------------------------------------------------

def build_augmentation_pipeline() -> A.Compose:
    """
    Return the training augmentation pipeline.

    Transforms chosen to simulate realistic in-car camera variation:
    - Brightness / contrast shift  → different times of day
    - Gaussian noise                → sensor noise
    - Gaussian blur                 → focus variation
    - Horizontal flip               → symmetrical eye appearance
    - Rotation + shift/scale        → head pose variation
    - Coarse dropout                → partial occlusion (glasses, glare)
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.GaussNoise(var_limit=(5, 25), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=6, max_width=6, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
    ])


AUGMENT_PIPELINE = build_augmentation_pipeline()


def augment_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    categories: list[str],
    augment_factor: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment the minority class to reduce class imbalance.

    Args:
        X:                uint8 array (N, H, W).
        Y:                int label array (N,).
        categories:       Class name list (for logging).
        augment_factor:   Target multiplier for minority class size.

    Returns:
        Shuffled (X_aug, Y_aug) arrays.
    """
    X_aug = list(X)
    Y_aug = list(Y)

    class_counts = np.bincount(Y)
    minority_cls = int(np.argmin(class_counts))
    minority_idx = np.where(Y == minority_cls)[0]

    logger.info(
        "Augmenting class '%s' (%d → target ~%d)",
        categories[minority_cls],
        len(minority_idx),
        len(minority_idx) * augment_factor,
    )

    for idx in tqdm(minority_idx, desc="Augmenting minority class"):
        img = X[idx]
        for _ in range(augment_factor - 1):
            aug_img = AUGMENT_PIPELINE(image=img)["image"]
            X_aug.append(aug_img)
            Y_aug.append(minority_cls)

    X_aug_arr = np.array(X_aug)
    Y_aug_arr = np.array(Y_aug)
    perm = np.random.permutation(len(X_aug_arr))
    return X_aug_arr[perm], Y_aug_arr[perm]


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def apply_smote(
    X: np.ndarray,
    Y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE in flattened pixel space.

    SMOTE interpolates between existing minority samples in feature
    space, providing genuine diversity vs. simple augmentation copies.

    Args:
        X:             uint8 (N, H, W) array.
        Y:             int label array.
        k_neighbors:   SMOTE neighbourhood size.
        random_state:  Reproducibility seed.

    Returns:
        (X_resampled, Y_resampled) — uint8 arrays.
    """
    from imblearn.over_sampling import SMOTE

    n_samples, h, w = X.shape
    X_flat = X.reshape(n_samples, -1).astype(np.float32) / 255.0

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, Y_res = smote.fit_resample(X_flat, Y)

    X_res = (X_res * 255).clip(0, 255).astype(np.uint8).reshape(-1, h, w)
    logger.info(
        "SMOTE: %d → %d samples | class dist: %s",
        n_samples, len(X_res), np.bincount(Y_res),
    )
    return X_res, Y_res
