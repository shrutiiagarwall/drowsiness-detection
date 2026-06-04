"""
src/utils/gradcam.py
======================
Grad-CAM explainability for CBAM-CNN.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def compute_gradcam(
    model: tf.keras.Model,
    tensor: np.ndarray,
    class_index: int | None = None,
    last_conv_layer_name: str = "conv3",
) -> np.ndarray:
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy().astype(np.float32)


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if original_image.ndim == 2 else original_image.copy()
    return cv2.addWeighted(base, 1 - alpha, heatmap_colored, alpha, 0)
