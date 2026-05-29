"""
src/utils/gradcam.py
======================
Grad-CAM explainability for the CBAM-CNN model.

Produces class activation maps that highlight which spatial regions
of the eye patch the network attended to when making its prediction.

Reference:
    Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization." ICCV 2017.
    https://arxiv.org/abs/1610.02391
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
    """
    Compute a Grad-CAM heatmap for the given input tensor.

    Args:
        model:                Trained Keras model.
        tensor:               Preprocessed input, shape (1, H, W, 1), float32.
        class_index:          Class to explain. If None, uses predicted class.
        last_conv_layer_name: Name of the last Conv2D layer in the model.

    Returns:
        Normalised heatmap array shape (H, W), float32 ∈ [0, 1].
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    heatmap_np = heatmap.numpy()
    return heatmap_np.astype(np.float32)


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image: uint8 grayscale or BGR image.
        heatmap:        float32 heatmap (H, W) from compute_gradcam.
        alpha:          Blend factor for heatmap overlay.
        colormap:       OpenCV colormap (default: JET).

    Returns:
        BGR uint8 overlay image.
    """
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    if original_image.ndim == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        base = original_image.copy()

    overlay = cv2.addWeighted(base, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
