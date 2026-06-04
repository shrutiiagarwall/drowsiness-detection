"""
src/preprocessing/augmentation.py
====================================
Augmentation pipelines for eye-state and yawn datasets.

Key fix in v2:
    augment_yawn_balanced() — augments BOTH classes to an equal target,
    instead of only augmenting the minority class.
    The original approach caused a ~3:1 imbalance because the yawn
    dataset was already nearly balanced (616 vs 617 images), so
    augmenting only the minority class created severe skew.
"""

from __future__ import annotations

import logging

import albumentations as A
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eye Augmentation Pipeline
# ---------------------------------------------------------------------------

EYE_AUGMENT_PIPELINE = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.GaussNoise(var_limit=(5, 25), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.4),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.CoarseDropout(max_holes=4, max_height=6, max_width=6, p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
])

# ---------------------------------------------------------------------------
# Yawn Augmentation Pipeline
# ---------------------------------------------------------------------------

YAWN_AUGMENT_PIPELINE = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.GaussNoise(var_limit=(5, 20), p=0.4),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.CoarseDropout(max_holes=3, max_height=8, max_width=8, p=0.2),
])


# ---------------------------------------------------------------------------
# Eye Dataset Augmentation (minority-class only)
# ---------------------------------------------------------------------------

def augment_eye_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    categories: list[str],
    augment_factor: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment the minority eye-state class to reduce imbalance.

    This is appropriate for the eye dataset where one class is
    genuinely under-represented.
    """
    X_aug = list(X)
    Y_aug = list(Y)

    class_counts = np.bincount(Y)
    minority_cls = int(np.argmin(class_counts))
    minority_idx = np.where(Y == minority_cls)[0]

    logger.info(
        "Augmenting eye class '%s' (%d → ~%d)",
        categories[minority_cls], len(minority_idx),
        len(minority_idx) * augment_factor,
    )

    for idx in tqdm(minority_idx, desc="Eye augmentation"):
        img = X[idx]
        for _ in range(augment_factor - 1):
            aug_img = EYE_AUGMENT_PIPELINE(image=img)["image"]
            X_aug.append(aug_img)
            Y_aug.append(minority_cls)

    X_arr = np.array(X_aug)
    Y_arr = np.array(Y_aug)
    perm = np.random.permutation(len(X_arr))
    return X_arr[perm], Y_arr[perm]


# ---------------------------------------------------------------------------
# Yawn Dataset Augmentation — BALANCED (v2 bug fix)
# ---------------------------------------------------------------------------

def augment_yawn_balanced(
    X: np.ndarray,
    Y: np.ndarray,
    categories: list[str],
    target_per_class: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment BOTH yawn classes to an equal target count.

    Why this matters (bug fix explanation):
        The yawn dataset contains ~616 no_yawn and ~617 yawn images —
        nearly balanced. The original code augmented only the minority
        class, producing 1851 no_yawn vs 617 yawn (a 3:1 imbalance).
        This function brings both classes to `target_per_class` samples,
        preserving balance and preventing the classifier from predicting
        the majority class most of the time.

    Args:
        X:                uint8 (N, H, W) images.
        Y:                int label array (N,).
        categories:       Class name list.
        target_per_class: How many samples each class should reach.

    Returns:
        Shuffled (X_aug, Y_aug).
    """
    X_out: list[np.ndarray] = []
    Y_out: list[int] = []
    counts = np.bincount(Y)
    logger.info(
        "Yawn balanced augmentation: %s → target %d each",
        {cat: int(c) for cat, c in zip(categories, counts)},
        target_per_class,
    )

    for cls in range(len(categories)):
        idx = np.where(Y == cls)[0]
        original = X[idx]
        n_orig = len(original)

        # Add all originals first
        X_out.extend(original)
        Y_out.extend([cls] * n_orig)

        # Augment up to target
        needed = target_per_class - n_orig
        if needed <= 0:
            logger.info("  Class '%s': already at target, no augmentation needed.", categories[cls])
            continue

        logger.info("  Augmenting '%s': %d → %d", categories[cls], n_orig, target_per_class)
        aug_count = 0
        while aug_count < needed:
            src = original[aug_count % n_orig]
            aug_img = YAWN_AUGMENT_PIPELINE(image=src)["image"]
            X_out.append(aug_img)
            Y_out.append(cls)
            aug_count += 1

    X_arr = np.array(X_out)
    Y_arr = np.array(Y_out)
    perm = np.random.permutation(len(X_arr))
    return X_arr[perm], Y_arr[perm]


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def apply_smote(
    X: np.ndarray,
    Y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling in flattened pixel space."""
    from imblearn.over_sampling import SMOTE

    n_samples, h, w = X.shape
    X_flat = X.reshape(n_samples, -1).astype(np.float32) / 255.0

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, Y_res = smote.fit_resample(X_flat, Y)

    X_res = (X_res * 255).clip(0, 255).astype(np.uint8).reshape(-1, h, w)
    logger.info(
        "SMOTE: %d → %d | class dist: %s",
        n_samples, len(X_res), np.bincount(Y_res),
    )
    return X_res, Y_res
