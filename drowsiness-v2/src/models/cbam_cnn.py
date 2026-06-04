"""
src/models/cbam_cnn.py
=======================
CBAM-CNN architecture — public, full implementation.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers


class ReduceMeanLayer(layers.Layer):
    def call(self, x):
        return keras.ops.mean(x, axis=-1, keepdims=True)
    def get_config(self):
        return super().get_config()


class ReduceMaxLayer(layers.Layer):
    def call(self, x):
        return keras.ops.max(x, axis=-1, keepdims=True)
    def get_config(self):
        return super().get_config()


def channel_attention(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    c = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(x)
    d1 = layers.Dense(c // ratio, activation="relu")
    d2 = layers.Dense(c, activation="sigmoid")
    avg_out = d2(d1(avg_pool))
    max_out = d2(d1(max_pool))
    return layers.Multiply()([x, layers.Add()([avg_out, max_out])])


def spatial_attention(x: tf.Tensor, kernel_size: int = 7) -> tf.Tensor:
    avg_out = ReduceMeanLayer()(x)
    max_out = ReduceMaxLayer()(x)
    concat = layers.Concatenate(axis=-1)([avg_out, max_out])
    scale = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([x, scale])


def cbam_block(x: tf.Tensor, ratio: int = 8, kernel_size: int = 7) -> tf.Tensor:
    x = channel_attention(x, ratio=ratio)
    x = spatial_attention(x, kernel_size=kernel_size)
    return x


def build_cbam_cnn(
    img_size: int = 48,
    num_classes: int = 2,
    conv_filters: tuple = (32, 64, 128),
    dropout_rates: tuple = (0.2, 0.3, 0.3, 0.4),
    l2_reg: float = 1e-4,
    cbam_ratio: int = 8,
    cbam_kernel: int = 7,
) -> Model:
    inp = keras.Input(shape=(img_size, img_size, 1), name="eye_input")
    x = inp
    for i, (filters, dropout) in enumerate(zip(conv_filters, dropout_rates)):
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu",
                          name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = cbam_block(x, ratio=cbam_ratio, kernel_size=cbam_kernel)
        x = layers.MaxPooling2D(2, name=f"pool{i+1}")(x)
        x = layers.Dropout(dropout, name=f"drop{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense_head")(x)
    x = layers.Dropout(dropout_rates[-1], name="drop_head")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="CBAM_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
