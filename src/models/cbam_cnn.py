"""
src/models/cbam_cnn.py
=======================
CBAM (Convolutional Block Attention Module) CNN architecture.

Architecture overview:
    Input → Conv Block 1 → CBAM → MaxPool → Dropout
          → Conv Block 2 → CBAM → MaxPool → Dropout
          → Conv Block 3 → CBAM → MaxPool → Dropout
          → GlobalAveragePooling
          → Dense(128) → Dropout
          → Dense(num_classes, softmax)

CBAM applies channel attention then spatial attention sequentially,
enabling the network to focus on the most informative feature maps
AND spatial locations — crucial for small eye-region inputs.

Reference:
    Woo et al. (2018). "CBAM: Convolutional Block Attention Module."
    ECCV 2018. https://arxiv.org/abs/1807.06521
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers


# ---------------------------------------------------------------------------
# Custom layers (Lambda-free — required for .keras serialisation)
# ---------------------------------------------------------------------------

class ReduceMeanLayer(layers.Layer):
    """Channel-wise mean reduction for spatial attention."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return keras.ops.mean(x, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        return super().get_config()


class ReduceMaxLayer(layers.Layer):
    """Channel-wise max reduction for spatial attention."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return keras.ops.max(x, axis=-1, keepdims=True)

    def get_config(self) -> dict:
        return super().get_config()


# ---------------------------------------------------------------------------
# Attention Blocks
# ---------------------------------------------------------------------------

def channel_attention(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """
    Channel Attention Module.

    Learns to emphasise informative feature channels (e.g., channels
    that encode iris texture vs. eyelid edges) using a shared MLP on
    both average-pooled and max-pooled channel descriptors.

    Args:
        x:     Input feature map (B, H, W, C).
        ratio: Bottleneck reduction ratio for the MLP.

    Returns:
        Channel-recalibrated feature map (same shape as x).
    """
    c = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(x)

    shared_dense_1 = layers.Dense(c // ratio, activation="relu")
    shared_dense_2 = layers.Dense(c, activation="sigmoid")

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    scale = layers.Add()([avg_out, max_out])
    return layers.Multiply()([x, scale])


def spatial_attention(x: tf.Tensor, kernel_size: int = 7) -> tf.Tensor:
    """
    Spatial Attention Module.

    After channel recalibration, spatial attention asks *where* the
    informative regions are by pooling across channels and learning a
    convolutional saliency map.

    Args:
        x:           Channel-attended feature map (B, H, W, C).
        kernel_size: Convolution kernel for the saliency map (default 7).

    Returns:
        Spatially-recalibrated feature map (same shape as x).
    """
    avg_out = ReduceMeanLayer()(x)
    max_out = ReduceMaxLayer()(x)
    concat = layers.Concatenate(axis=-1)([avg_out, max_out])  # (B, H, W, 2)
    scale = layers.Conv2D(
        1, kernel_size, padding="same", activation="sigmoid"
    )(concat)
    return layers.Multiply()([x, scale])


def cbam_block(x: tf.Tensor, ratio: int = 8, kernel_size: int = 7) -> tf.Tensor:
    """Apply CBAM: channel attention followed by spatial attention."""
    x = channel_attention(x, ratio=ratio)
    x = spatial_attention(x, kernel_size=kernel_size)
    return x


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_cbam_cnn(
    img_size: int = 48,
    num_classes: int = 2,
    conv_filters: tuple[int, ...] = (32, 64, 128),
    dropout_rates: tuple[float, ...] = (0.2, 0.3, 0.3, 0.4),
    l2_reg: float = 1e-4,
    cbam_ratio: int = 8,
    cbam_kernel: int = 7,
) -> Model:
    """
    Build the CBAM-augmented CNN model.

    Args:
        img_size:      Square input resolution (e.g., 48 for eye patches).
        num_classes:   Output classes (2 for Closed/Open).
        conv_filters:  Filter counts for each Conv block.
        dropout_rates: Dropout rates after each block + final dense layer.
        l2_reg:        L2 weight decay for dense layers.
        cbam_ratio:    Channel attention bottleneck ratio.
        cbam_kernel:   Spatial attention kernel size.

    Returns:
        Compiled Keras Model (not yet trained).
    """
    inp = keras.Input(shape=(img_size, img_size, 1), name="eye_input")

    x = inp
    for i, (filters, dropout) in enumerate(zip(conv_filters, dropout_rates)):
        x = layers.Conv2D(
            filters, (3, 3), padding="same", activation="relu",
            name=f"conv{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = cbam_block(x, ratio=cbam_ratio, kernel_size=cbam_kernel)
        x = layers.MaxPooling2D(2, name=f"pool{i+1}")(x)
        x = layers.Dropout(dropout, name=f"drop{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_head"
    )(x)
    x = layers.Dropout(dropout_rates[-1], name="drop_head")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="CBAM_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
