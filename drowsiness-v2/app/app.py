"""
app/app.py
===========
DrowsyGuard v2 — Gradio Interface

Bug fixes applied vs original notebook:
  1. `faces` variable defined BEFORE fallback yawn block (NameError fix)
  2. PERCLOS uses `and` not `or` (both eyes must be closed)
  3. Mouth detection: minNeighbors=3 (was 5 — missed real yawns)
  4. Timeline plot: scatter fallback for single-frame sessions + xlim guard
  5. Cascade URLs: mouth cascade from 2.4.13.7 branch, not broken 2.4
  6. Demo mode: graceful mock when either weight file is absent
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference_engine import (
    load_model_or_mock, MockEyeModel, MockYawnModel,
    prep_eye, prep_mouth,
)
from src.features.perclos import FusionDrowsinessScorer
from src.utils.cascade_utils import load_cascades

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE      = 48
YAWN_IMG_SIZE = 64
WEIGHTS_EYE   = ROOT / "models" / "weights" / "cbam_cnn_final.keras"
WEIGHTS_YAWN  = ROOT / "models" / "weights" / "yawn_model_final.keras"
CASCADE_DIR   = ROOT / "cascades"

STATUS_COLORS = {"OK": "#27ae60", "WARNING": "#f39c12", "ALERT": "#e74c3c"}
STATUS_ICONS  = {"OK": "✅",       "WARNING": "⚠️",       "ALERT": "🚨"}

# ---------------------------------------------------------------------------
# Load models (demo mode if weights absent)
# ---------------------------------------------------------------------------
eye_model  = load_model_or_mock(WEIGHTS_EYE,  mock_class=MockEyeModel)
yawn_model = load_model_or_mock(WEIGHTS_YAWN, mock_class=MockYawnModel)
IS_DEMO    = isinstance(eye_model, MockEyeModel) or isinstance(yawn_model, MockYawnModel)

# Load cascades (downloads if missing)
try:
    cascades = load_cascades(CASCADE_DIR)
    CASCADES_OK = True
except Exception as e:
    logger.warning("Cascades not loaded: %s — multi-region detection disabled.", e)
    cascades = {}
    CASCADES_OK = False

# Session state
_fusion = FusionDrowsinessScorer()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def analyze_frame(image_input: Image.Image | None):
    global _fusion

    if image_input is None:
        return None, "⚠️ Upload an image or use your webcam.", None

    try:
        frame = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        h, w  = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font  = cv2.FONT_HERSHEY_SIMPLEX

        l_state = r_state = 1   # default: open
        yawning = False

        # ── Face detection (FIX: defined before fallback yawn block) ──────
        if CASCADES_OK:
            faces = cascades["face"].detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        else:
            faces = []

        # Draw face rects
        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (130, 130, 130), 1)

        # ── Eye detection ──────────────────────────────────────────────────
        if CASCADES_OK:
            for (x, y, ew, eh) in cascades["left_eye"].detectMultiScale(gray):
                roi = frame[y:y+eh, x:x+ew]
                if roi.size == 0:
                    break
                p = eye_model.predict(prep_eye(roi, IMG_SIZE), verbose=0)
                l_state = int(np.argmax(p[0]))
                clr = (0, 0, 255) if l_state == 0 else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+ew, y+eh), clr, 2)
                cv2.putText(frame, "Closed" if l_state == 0 else "Open",
                            (x, y - 4), font, 0.4, clr, 1)
                break

            for (x, y, ew, eh) in cascades["right_eye"].detectMultiScale(gray):
                roi = frame[y:y+eh, x:x+ew]
                if roi.size == 0:
                    break
                p = eye_model.predict(prep_eye(roi, IMG_SIZE), verbose=0)
                r_state = int(np.argmax(p[0]))
                clr = (0, 0, 255) if r_state == 0 else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+ew, y+eh), clr, 2)
                cv2.putText(frame, "Closed" if r_state == 0 else "Open",
                            (x, y - 4), font, 0.4, clr, 1)
                break
        else:
            # No cascades: run model on full image as eye patch
            tensor = prep_eye(frame, IMG_SIZE)
            p = eye_model.predict(tensor, verbose=0)
            l_state = r_state = int(np.argmax(p[0]))

        # ── Yawn detection ──────────────────────────────────────────────────
        if CASCADES_OK:
            lower_gray = gray[h // 2:, :]
            # FIX: minNeighbors=3 (was 5 — too strict, missed real yawns)
            mouths = cascades["mouth"].detectMultiScale(
                lower_gray, scaleFactor=1.1,
                minNeighbors=3, minSize=(25, 15)
            )

            if len(mouths) > 0:
                for (mx, my, mw, mh) in mouths:
                    ry = my + h // 2
                    roi = frame[ry:ry+mh, mx:mx+mw]
                    if roi.size == 0:
                        break
                    p = yawn_model.predict(prep_mouth(roi, YAWN_IMG_SIZE), verbose=0)
                    yawning = (np.argmax(p[0]) == 1)
                    clr = (0, 0, 255) if yawning else (0, 255, 255)
                    cv2.rectangle(frame, (mx, ry), (mx+mw, ry+mh), clr, 2)
                    cv2.putText(frame, "YAWN" if yawning else "No Yawn",
                                (mx, ry - 5), font, 0.45, clr, 1)
                    break
            else:
                # ── Fallback: use face ROI lower 40% as mouth region ──────
                # FIX: `faces` was not defined here in original notebook
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    my1 = fy + int(fh * 0.60)
                    my2 = fy + int(fh * 0.95)
                    mx1 = fx + int(fw * 0.20)
                    mx2 = fx + int(fw * 0.80)
                    mouth_roi = gray[my1:my2, mx1:mx2]
                    if mouth_roi.size > 0:
                        mri = cv2.resize(mouth_roi, (YAWN_IMG_SIZE, YAWN_IMG_SIZE))
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                        mri = clahe.apply(mri).astype("float32") / 255.0
                        p = yawn_model.predict(
                            mri.reshape(1, YAWN_IMG_SIZE, YAWN_IMG_SIZE, 1), verbose=0
                        )
                        yawning = (np.argmax(p[0]) == 1)
                        clr = (0, 0, 255) if yawning else (0, 255, 255)
                        cv2.rectangle(frame, (mx1, my1), (mx2, my2), clr, 2)
                        cv2.putText(frame,
                                    "YAWN(F)" if yawning else "No Yawn(F)",
                                    (mx1, my1 - 5), font, 0.45, clr, 1)

        # ── Fusion — FIX: AND not OR ──────────────────────────────────────
        eye_closed = (l_state == 0) and (r_state == 0)
        res = _fusion.update(eye_closed, yawning)
        score  = res["fusion_score"]
        status = res["status"]
        clr    = tuple(int(c * 255) for c in
                       plt.cm.RdYlGn(1 - score)[:3])[::-1]   # BGR
        sclr   = {"OK": (0, 220, 80), "WARNING": (0, 165, 255), "ALERT": (0, 0, 255)}[status]

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"Status: {status}", (10, h - 46), font, 0.7, sclr, 2)
        cv2.putText(frame,
                    f"PERCLOS:{res['perclos']*100:.0f}%  Yawn:{res['yawn_rate']*100:.0f}%",
                    (10, h - 24), font, 0.5, (180, 180, 180), 1)
        cv2.putText(frame,
                    f"Score:{score:.2f}  Yawns:{res['yawn_count']}",
                    (10, h - 6), font, 0.5, (180, 180, 180), 1)

        bx, by, bw2, bh2 = w - 150, 10, 130, 16
        cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw2 * score), by + bh2), sclr, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), (200, 200, 200), 1)
        cv2.putText(frame, f"Score {score:.2f}", (bx, by + bh2 + 13), font, 0.42, (200, 200, 200), 1)

        if status == "ALERT":
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), min(20, 4 + int(score * 20)))
        elif status == "WARNING":
            cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 3)

        # ── Status text ───────────────────────────────────────────────────
        emj = {"OK": "✅", "WARNING": "⚠️", "ALERT": "🚨"}
        txt = (
            f"{emj[status]} **{status}**\n\n"
            f"🔵 Fusion Score : `{score:.3f}`\n"
            f"👁️ PERCLOS      : `{res['perclos']*100:.1f}%`\n"
            f"😮 Yawn Rate    : `{res['yawn_rate']*100:.1f}%`\n"
            f"💤 Yawn Count   : `{res['yawn_count']}`\n"
            f"👁️ Left Eye     : `{'Closed' if l_state==0 else 'Open'}`\n"
            f"👁️ Right Eye    : `{'Closed' if r_state==0 else 'Open'}`\n"
            f"😮 Yawning      : `{'Yes' if yawning else 'No'}`\n"
        )
        if IS_DEMO:
            txt += "\n> ⚠️ **DEMO MODE** — mock predictions (weights not loaded)"

        # ── Timeline plot — FIX: scatter fallback + xlim guard ───────────
        sc = _fusion.score_history[-120:]
        n  = len(sc)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        ax.axhspan(0.65, 1.0,  alpha=0.15, color="red")
        ax.axhspan(0.45, 0.65, alpha=0.15, color="orange")
        ax.axhspan(0.0,  0.45, alpha=0.10, color="green")

        if n >= 2:
            ax.plot(range(n), sc, color="#00d4ff", linewidth=1.5)
            ax.fill_between(range(n), sc, alpha=0.2, color="#00d4ff")
        else:
            # FIX: single-point fallback (plot crashes on n<2 with line)
            ax.scatter([0], sc if sc else [0], color="#00d4ff", s=30, zorder=5)

        ax.axhline(0.65, color="red",    linestyle="--", linewidth=1, alpha=0.7, label="Alert")
        ax.axhline(0.45, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Warn")
        # FIX: xlim guard — max(n, 2) prevents zero-width x-axis
        ax.set_xlim(-0.5, max(n, 2))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frames", color="white", fontsize=8)
        ax.set_ylabel("Score",  color="white", fontsize=8)
        ax.set_title("Fusion Score Timeline", color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(fontsize=7, facecolor="#16213e", labelcolor="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), txt, Image.open(buf)

    except Exception as e:
        import traceback
        return None, f"❌ Error:\n```\n{traceback.format_exc()}\n```", None


def reset_session():
    global _fusion
    _fusion = FusionDrowsinessScorer()
    return "🔄 Session reset."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="DrowsyGuard — Driver Drowsiness Detection",
    theme=gr.themes.Base(primary_hue="blue").set(
        body_background_fill="#1a1a2e",
        body_background_fill_dark="#1a1a2e",
        block_background_fill="#16213e",
        block_background_fill_dark="#16213e",
        block_border_color="#0f3460",
    ),
    css=".gradio-container{max-width:980px;margin:auto}",
) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 4px">
      <h1 style="font-size:2em;color:#7eb3ff;margin:0">🚗 DrowsyGuard v2</h1>
      <p style="color:#888;margin-top:6px">
        Multi-Signal Driver Drowsiness Detection &nbsp;·&nbsp;
        CBAM-CNN + Yawn Fusion &nbsp;·&nbsp; Research Prototype
      </p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🔍 Inference"):
            gr.Markdown(
                "Upload a **face or eye image** (webcam or file). "
                "The system detects eye state and yawning, then computes "
                "PERCLOS-based drowsiness score in real time."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(
                        sources=["webcam", "upload"],
                        type="pil",
                        label="📷 Camera / Upload",
                        mirror_webcam=True,
                        height=320,
                    )
                    with gr.Row():
                        btn_a = gr.Button("🔍 Analyze Frame", variant="primary")
                        btn_r = gr.Button("🔄 Reset Session", variant="secondary")
                    rmsg = gr.Textbox(show_label=False, interactive=False)

                with gr.Column(scale=1):
                    out_f = gr.Image(label="📸 Annotated Output", height=320)

            with gr.Row():
                with gr.Column(scale=1):
                    out_s = gr.Markdown(value="_Waiting for first frame…_")
                with gr.Column(scale=2):
                    out_p = gr.Image(label="📈 Fusion Score Timeline", height=200)

            btn_a.click(fn=analyze_frame, inputs=[inp], outputs=[out_f, out_s, out_p])
            btn_r.click(fn=reset_session, inputs=[], outputs=[rmsg])

            if IS_DEMO:
                gr.HTML("""
                <div style="background:#f39c1218;border:1px solid #f39c12;
                            border-radius:6px;padding:10px 14px;color:#f39c12;
                            font-family:sans-serif;font-size:0.85em;margin-top:8px;">
                  ⚠️ <strong>DEMO MODE</strong> — Model weights not present.
                  Predictions are illustrative only. See README for access.
                </div>
                """)

        with gr.Tab("🐛 Bug Fixes (v2)"):
            gr.Markdown("""
## Bug Fixes Applied in This Version

| # | Bug | Fix |
|---|-----|-----|
| 1 | `faces` NameError in fallback yawn block | Defined before block |
| 2 | PERCLOS used `OR` — single closed eye triggered alert | Fixed to `AND` |
| 3 | `minNeighbors=5` missed real yawns | Lowered to `3` |
| 4 | Timeline crashed on 1-frame session | Scatter fallback + xlim guard |
| 5 | Mouth cascade 404 (wrong branch URL) | Using `2.4.13.7` tag |
| 6 | Yawn augmentation created 3:1 imbalance | Both classes augmented equally |
| 7 | Duplicate `MODELS_DIR` definition | Removed duplicate |
| 8 | Dataset path double-nested | `extract_to='/content'` |
            """)

        with gr.Tab("📄 About"):
            gr.Markdown("""
## About

**DrowsyGuard v2** is a research prototype for a pending academic publication
on multi-signal driver drowsiness detection.

### Model Weights

Model weights and pre-trained artefacts are **not included** in this repository.
They are available **upon request for research collaboration**.

Contact: `[your-email@institution.edu]`

### License

MIT License — Novel feature engineering methodology subject to separate IP disclosure.
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
