"""
app/app.py  (also symlinked as root-level app.py for HuggingFace Spaces)
=========================================================================
DrowsyGuard — Gradio Web Interface

Features
--------
- Upload a cropped eye image → get drowsiness prediction + confidence bar
- Graceful DEMO mode when model weights are absent
- Clear visual alert levels (OK / WARNING / ALERT)
- System architecture overview tab
- About tab with research disclosure
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference_engine import (
    load_model_or_mock,
    preprocess_pil,
    run_inference,
)
from src.features.perclos import FusionDrowsinessScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_PATH = ROOT / "models" / "weights" / "cbam_cnn_final.keras"
IMG_SIZE = 48

STATUS_COLORS = {
    "OK":      "#27ae60",
    "WARNING": "#f39c12",
    "ALERT":   "#e74c3c",
}
STATUS_ICONS = {
    "OK":      "✅",
    "WARNING": "⚠️",
    "ALERT":   "🚨",
}

# ---------------------------------------------------------------------------
# Model & Session State (loaded once at startup)
# ---------------------------------------------------------------------------

model = load_model_or_mock(WEIGHTS_PATH)
IS_DEMO = hasattr(model, "name") and model.name == "MockModel"
fusion_scorer = FusionDrowsinessScorer()

# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------

def predict_eye_image(pil_image: Image.Image | None):
    """
    Main prediction callback for the Gradio interface.

    Args:
        pil_image: PIL Image uploaded by the user.

    Returns:
        Tuple of Gradio component updates.
    """
    # ── Input validation ─────────────────────────────────────────────────
    if pil_image is None:
        return (
            _status_html("—", "Upload an image to begin.", "#555"),
            _confidence_html(None, None),
            _metrics_html(None),
            _demo_banner(),
        )

    start = time.perf_counter()

    try:
        tensor = preprocess_pil(pil_image, img_size=IMG_SIZE)
    except Exception as e:
        return (
            _status_html("ERROR", f"Could not process image: {e}", "#c0392b"),
            _confidence_html(None, None),
            _metrics_html(None),
            _demo_banner(),
        )

    result = run_inference(model, tensor)
    latency_ms = (time.perf_counter() - start) * 1000

    # Update temporal fusion scorer
    fusion_result = fusion_scorer.update(
        eyes_closed=result.is_drowsy, yawning=False  # single-frame: no yawn signal
    )

    status = fusion_result.status
    color = STATUS_COLORS[status]
    icon = STATUS_ICONS[status]

    return (
        _status_html(
            f"{icon} {status}",
            f"Eye State: <strong>{result.label_name}</strong> "
            f"— Confidence: {result.confidence * 100:.1f}%  "
            f"({latency_ms:.0f} ms)",
            color,
        ),
        _confidence_html(result.label_name, result.confidence),
        _metrics_html(fusion_result),
        _demo_banner(),
    )


def reset_session():
    """Reset the temporal fusion scorer between driving sessions."""
    fusion_scorer.reset()
    return (
        _status_html("—", "Session reset. Upload a new image.", "#555"),
        _confidence_html(None, None),
        _metrics_html(None),
        _demo_banner(),
    )


# ---------------------------------------------------------------------------
# HTML Component Builders
# ---------------------------------------------------------------------------

def _status_html(title: str, subtitle: str, color: str) -> str:
    return f"""
    <div style="
        background: {color}18;
        border-left: 5px solid {color};
        border-radius: 8px;
        padding: 16px 20px;
        font-family: 'Segoe UI', sans-serif;
    ">
      <div style="font-size: 1.5em; font-weight: 700; color: {color};">
        {title}
      </div>
      <div style="margin-top: 6px; color: #ccc; font-size: 0.95em;">
        {subtitle}
      </div>
    </div>
    """


def _confidence_html(label: str | None, confidence: float | None) -> str:
    if label is None or confidence is None:
        return "<p style='color:#888; font-family:sans-serif;'>No prediction yet.</p>"

    pct = confidence * 100
    color = STATUS_COLORS["ALERT"] if label == "Closed" else STATUS_COLORS["OK"]

    return f"""
    <div style="font-family: 'Segoe UI', sans-serif; padding: 8px 0;">
      <div style="margin-bottom: 6px; font-weight: 600; color: #ddd;">
        {label} — {pct:.1f}%
      </div>
      <div style="
        background: #333;
        border-radius: 999px;
        height: 14px;
        overflow: hidden;
      ">
        <div style="
          width: {pct:.1f}%;
          background: {color};
          height: 100%;
          border-radius: 999px;
          transition: width 0.4s ease;
        "></div>
      </div>
    </div>
    """


def _metrics_html(fusion_result) -> str:
    if fusion_result is None:
        return "<p style='color:#888; font-family:sans-serif;'>—</p>"
    perclos_pct = fusion_result.perclos * 100
    yawn_pct = fusion_result.yawn_rate * 100
    score = fusion_result.fusion_score
    return f"""
    <div style="font-family: monospace; font-size: 0.9em; color: #bbb; line-height: 1.9;">
      <div>🎯 PERCLOS  : <strong>{perclos_pct:.1f}%</strong></div>
      <div>😮 Yawn Rate: <strong>{yawn_pct:.1f}%</strong></div>
      <div>⚡ Fusion   : <strong>{score:.3f}</strong></div>
      <div>😴 Yawns    : <strong>{fusion_result.yawn_count}</strong></div>
    </div>
    """


def _demo_banner() -> str:
    if IS_DEMO:
        return """
        <div style="
            background: #f39c1218;
            border: 1px solid #f39c12;
            border-radius: 6px;
            padding: 10px 14px;
            color: #f39c12;
            font-family: sans-serif;
            font-size: 0.85em;
        ">
          ⚠️ <strong>DEMO MODE</strong> — Model weights not found.
          Predictions are illustrative only.
          See README for weight access information.
        </div>
        """
    return "<div></div>"


# ---------------------------------------------------------------------------
# Architecture Diagram (ASCII — always visible)
# ---------------------------------------------------------------------------

ARCHITECTURE_MD = """
## System Architecture

```
Input Image
    │
    ▼
┌─────────────────────────┐
│  CLAHE Preprocessing    │   Contrast normalisation for illumination robustness
└────────────┬────────────┘
             │
    ┌────────▼────────┐
    │  CBAM-CNN       │   Convolutional Block Attention Module
    │  32→64→128 feat │   Channel + Spatial attention heads
    └────────┬────────┘
             │  (also: MobileNetV2 + EfficientNetB0 in ensemble)
             │
    ┌────────▼────────┐
    │  Soft-Vote       │   3-model ensemble, uniform weights
    │  Ensemble        │
    └────────┬────────┘
             │
    ┌────────▼────────────────┐
    │  Temporal Fusion Scorer │   PERCLOS × 0.6 + Yawn × 0.4
    │  (90-frame window)      │   Sliding window, ~3 s @ 30 fps
    └────────┬────────────────┘
             │
    ┌────────▼────────┐
    │  Alert Engine   │   OK / WARNING (>0.45) / ALERT (>0.65)
    └─────────────────┘
```

**Novel Contributions (see paper for details)**
- CBAM attention adapted for small ocular patches (48 × 48 px)
- Custom temporal smoothing transform (proprietary; available on request)
- Weighted fusion calibrated against PERCLOS clinical benchmarks
"""

ABOUT_MD = """
## About

**DrowsyGuard** is a research prototype developed as part of a pending
academic publication on multi-signal driver drowsiness detection.

### Intellectual Property

The data pipeline, preprocessing code, model architecture, and UI are
fully open-source under the MIT License.

**Model weights and pre-trained artefacts are available upon request
for research collaboration.** The novel feature engineering methodology
(Section 3.2 of the paper) is subject to a separate IP disclosure and
is not included in this public release.

To request research access, contact: `[your-email@institution.edu]`

### Dataset

Experiments used a custom-curated dataset of eye-state (Closed/Open)
and mouth-state (yawn/no_yawn) images. No dataset is bundled with this
repository.

### Acknowledgements

> *[Add funding body, institution, and collaborator acknowledgements here.]*
"""

# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="DrowsyGuard — Driver Drowsiness Detection",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
    ).set(
        body_background_fill="#1a1a2e",
        body_background_fill_dark="#1a1a2e",
        block_background_fill="#16213e",
        block_background_fill_dark="#16213e",
        block_border_color="#0f3460",
    ),
    css="""
        .gradio-container { max-width: 960px; margin: auto; }
        h1, h2, h3 { color: #e0e0ff; }
        .gr-button-primary { background: #0f3460 !important; }
    """,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 8px 0;">
      <h1 style="font-size: 2em; color: #7eb3ff; margin: 0;">
        🚗 DrowsyGuard
      </h1>
      <p style="color: #888; margin-top: 6px;">
        Multi-Signal Driver Drowsiness Detection &nbsp;|&nbsp;
        Research Prototype &nbsp;|&nbsp; Pending Publication
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Inference ──────────────────────────────────────────────
        with gr.Tab("🔍 Inference"):
            gr.Markdown(
                "Upload a **cropped eye image** (grayscale or colour). "
                "The model will classify the eye state and compute the "
                "PERCLOS-based drowsiness score."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Eye Image Input",
                        height=220,
                    )
                    with gr.Row():
                        predict_btn = gr.Button(
                            "▶  Run Detection", variant="primary"
                        )
                        reset_btn = gr.Button("↺  Reset Session")

                with gr.Column(scale=1):
                    status_output = gr.HTML(
                        value=_status_html("—", "Upload an image to begin.", "#555"),
                        label="Status",
                    )
                    confidence_output = gr.HTML(
                        value=_confidence_html(None, None),
                        label="Confidence",
                    )
                    metrics_output = gr.HTML(
                        value=_metrics_html(None),
                        label="Temporal Metrics",
                    )
                    demo_banner_output = gr.HTML(value=_demo_banner())

            predict_btn.click(
                fn=predict_eye_image,
                inputs=[image_input],
                outputs=[
                    status_output,
                    confidence_output,
                    metrics_output,
                    demo_banner_output,
                ],
            )
            reset_btn.click(
                fn=reset_session,
                inputs=[],
                outputs=[
                    status_output,
                    confidence_output,
                    metrics_output,
                    demo_banner_output,
                ],
            )

            gr.Examples(
                examples=[
                    [str(p)]
                    for p in sorted(
                        (ROOT / "data" / "samples").glob("*.jpg")
                    )
                ],
                inputs=[image_input],
                label="Sample Images",
                examples_per_page=6,
            )

        # ── Tab 2: Architecture ───────────────────────────────────────────
        with gr.Tab("🏗️ Architecture"):
            gr.Markdown(ARCHITECTURE_MD)

        # ── Tab 3: About ──────────────────────────────────────────────────
        with gr.Tab("📄 About & Citation"):
            gr.Markdown(ABOUT_MD)

    # ── Footer ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 16px 0 4px 0;
                color: #555; font-size: 0.8em;">
      DrowsyGuard &nbsp;·&nbsp; MIT License &nbsp;·&nbsp;
      Model weights available upon research request
    </div>
    """)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
