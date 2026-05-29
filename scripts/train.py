"""
scripts/train.py
=================
Command-line training script.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    logger.info("Config loaded from %s", config_path)

    # ── Imports ────────────────────────────────────────────────────────────
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,
    )

    from src.preprocessing.loader import load_dataset
    from src.preprocessing.augmentation import augment_dataset, apply_smote
    from src.preprocessing.tensors import prepare_tensors, make_tf_dataset
    from src.models.cbam_cnn import build_cbam_cnn
    from src.models.transfer_models import build_mobilenet, build_efficientnet
    from src.models.ensemble import SoftVotingEnsemble

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    DATASET_DIR = Path(data_cfg["dataset_dir"])
    TRAIN_DIR = DATASET_DIR / data_cfg["train_subdir"]
    TEST_DIR = DATASET_DIR / data_cfg["test_subdir"]
    CATEGORIES = data_cfg["categories"]
    IMG_SIZE = data_cfg["img_size"]
    WEIGHTS_DIR = Path(cfg["paths"]["weights_dir"])
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(data_cfg["random_seed"])
    tf.random.set_seed(data_cfg["random_seed"])

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading training data…")
    X_train_raw, Y_train = load_dataset(TRAIN_DIR, CATEGORIES, IMG_SIZE)
    logger.info("Loading test data…")
    X_test_raw, Y_test = load_dataset(TEST_DIR, CATEGORIES, IMG_SIZE)

    # ── Augmentation ───────────────────────────────────────────────────────
    X_aug, Y_aug = augment_dataset(
        X_train_raw, Y_train, CATEGORIES,
        augment_factor=cfg["preprocessing"]["augment_factor"],
    )
    if cfg["preprocessing"]["apply_smote"]:
        counts = np.bincount(Y_aug)
        if counts.max() / counts.min() > 1.3:
            X_aug, Y_aug = apply_smote(
                X_aug, Y_aug,
                k_neighbors=cfg["preprocessing"]["smote_k_neighbors"],
            )

    # ── Tensors ────────────────────────────────────────────────────────────
    X_tr, Y_tr = prepare_tensors(X_aug, Y_aug)
    X_te, Y_te = prepare_tensors(X_test_raw, Y_test)
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_tr, Y_tr, test_size=data_cfg["val_split"],
        random_state=data_cfg["random_seed"], stratify=Y_tr,
    )
    train_ds = make_tf_dataset(X_tr, Y_tr, train_cfg["batch_size"])
    val_ds = make_tf_dataset(X_val, Y_val, train_cfg["batch_size"], shuffle=False)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(Y_tr), y=Y_tr
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info("Class weights: %s", class_weight_dict)

    # ── Build & Train CBAM-CNN ─────────────────────────────────────────────
    cbam_model = build_cbam_cnn(
        img_size=IMG_SIZE,
        num_classes=model_cfg["num_classes"],
        l2_reg=train_cfg["l2_regularization"],
        cbam_ratio=model_cfg["cbam_ratio"],
        cbam_kernel=model_cfg["cbam_kernel"],
    )
    cbam_model.summary()

    callbacks = [
        EarlyStopping(patience=train_cfg["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(WEIGHTS_DIR / "cbam_cnn_best.keras"),
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=train_cfg["reduce_lr_factor"],
                          patience=train_cfg["reduce_lr_patience"],
                          verbose=1),
        TensorBoard(log_dir=str(ROOT / cfg["paths"]["logs_dir"])),
    ]

    cbam_model.fit(
        train_ds, validation_data=val_ds,
        epochs=train_cfg["epochs"],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    cbam_model.save(str(WEIGHTS_DIR / "cbam_cnn_final.keras"))
    logger.info("✅ CBAM-CNN saved.")

    # ── Evaluate ───────────────────────────────────────────────────────────
    test_ds = make_tf_dataset(X_te, Y_te, train_cfg["batch_size"], shuffle=False)
    loss, acc = cbam_model.evaluate(test_ds, verbose=0)
    logger.info("Test accuracy: %.4f | Loss: %.4f", acc, loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DrowsyGuard models.")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML."
    )
    args = parser.parse_args()
    main(args.config)
