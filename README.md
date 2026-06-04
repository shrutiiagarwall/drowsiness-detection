# 🚗 DrowsyGuard: Real-Time Multi-Signal Driver Drowsiness Detection

<p align="center">
  <img src="assets/banner.png" alt="DrowsyGuard Banner" width="800"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Gradio-4.x-purple"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Pending%20Submission-red"/></a>
</p>

---

## 📄 Abstract

> *[Placeholder — to be updated upon journal submission.]*
>
> Driver drowsiness remains one of the leading causes of road fatalities globally.
> This work presents **DrowsyGuard**, a multi-signal ensemble framework that fuses
> eye-state classification (PERCLOS metric) with yawn detection via a weighted
> temporal fusion score. The pipeline introduces a **CBAM (Convolutional Block
> Attention Module)** backbone, a **balanced yawn augmentation strategy** that
> prevents class inversion, a **cascade-fallback mouth detection** path for
> robustness, and a **Grad-CAM explainability** layer for interpretability.
> All components have been validated and bug-fixed against 8 identified failure
> modes, including incorrect PERCLOS logic (`or` → `and`), cascade URL breakage,
> and face-variable scope errors in the Gradio inference path.

---

## 🐛 Bug Fixes in This Version (v2)

| # | Bug | Fix |
|---|-----|-----|
| 1 | Dataset path double-nesting (`/dataset_new/dataset_new`) | Fixed `extract_to='/content'` |
| 2 | `faces` NameError in Gradio fallback yawn block | Added `faces = _face_cc.detectMultiScale(...)` before usage |
| 3 | Mouth cascade broken URL (`opencv/2.4` branch missing file) | Switched to `2.4.13.7` branch for `haarcascade_mcs_mouth.xml` |
| 4 | `minNeighbors=5` too strict → missed real yawns | Lowered to `minNeighbors=3` |
| 5 | `eye_closed = l or r` wrong PERCLOS | Fixed to `and` — both eyes must be closed |
| 6 | Yawn augmentation created 3:1 imbalance (only minority augmented) | `augment_yawn_balanced()` — both classes augmented to equal target |
| 7 | Duplicate `MODELS_DIR` / `Path` import in yawn cell | Removed duplicate, reused existing variable |
| 8 | Fusion score timeline plot crashed on single-frame session | Added `scatter` fallback + `xlim` guard |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                         │
│     Webcam Frame / Uploaded Image / REST Payload        │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  Haar Cascade Detection   │
         │  Face + Left/Right Eyes   │
         │  + Mouth (with fallback)  │  ← NEW: face-crop fallback
         └──────┬────────────┬───────┘
                │            │
     ┌──────────▼──┐   ┌─────▼────────────┐
     │  Eye Region │   │  Mouth Region    │
     │  (48×48 px) │   │  (64×64 px)      │
     └──────┬──────┘   └──────┬───────────┘
            │                 │
┌───────────▼──────┐  ┌───────▼──────────────┐
│  CBAM-CNN        │  │  Yawn CNN Classifier  │
│  Ensemble        │  │  (Balanced training)  │  ← FIXED: balanced aug
│  Soft-Voting     │  │                       │
└───────────┬──────┘  └───────┬───────────────┘
            │                 │
    ┌────────▼─────────────────▼──────┐
    │     Temporal Fusion Scorer      │
    │  PERCLOS(both eyes) × 0.6       │  ← FIXED: `and` not `or`
    │  + Yawn Rate × 0.4              │
    │  Sliding window (90 frames)     │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  Alert Decision Engine      │
    │  OK / WARNING / ALERT       │
    └─────────────────────────────┘
```

---

## 📁 Repository Structure

```
drowsiness-detection/
│
├── 📓 notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_and_xai.ipynb
│   └── 04_drift_detection.ipynb
│
├── 🔧 src/
│   ├── preprocessing/
│   │   ├── loader.py               # CLAHE, eye + yawn dataset loading
│   │   ├── augmentation.py         # Albumentations + balanced SMOTE
│   │   └── tensors.py              # TF dataset builders
│   ├── features/
│   │   ├── feature_extractor.py    # Public stub (IP-protected)
│   │   └── perclos.py              # PERCLOS + fusion scorer (FIXED: and logic)
│   ├── models/
│   │   ├── cbam_cnn.py             # CBAM architecture
│   │   ├── transfer_models.py      # MobileNetV2 / EfficientNetB0
│   │   └── ensemble.py             # Soft-voting ensemble
│   └── utils/
│       ├── gradcam.py              # Grad-CAM explainability
│       ├── drift_detector.py       # ADWIN drift wrapper
│       └── cascade_utils.py        # Cascade downloader with fallback URLs
│
├── 🖥️  app/
│   ├── app.py                      # Gradio UI (FIXED: faces var, fallback yawn)
│   ├── api.py                      # FastAPI REST server
│   └── inference_engine.py         # Model loader + mock fallback
│
├── ⚙️  configs/
│   └── config.yaml
│
├── 🧪 tests/
├── 📜 scripts/
│   ├── train.py
│   └── download_cascades.py        # NEW: standalone cascade downloader
│
├── app.py                          # HuggingFace Spaces entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/shrutiiagarwall/drowsiness-detection.git
cd drowsiness-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download Haar cascades (required for inference):

```bash
python scripts/download_cascades.py
```

---

## 🚀 Quick Start

```bash
python app.py          # Gradio UI → http://localhost:7860
```

Or REST API:

```bash
uvicorn app.api:app --port 8000
```

---

## 🤖 Model Weights & Pre-trained Artifacts

> **⚠️ Model weights and pre-trained artifacts are not included in this public repository.**
>
> Trained CBAM-CNN, balanced Yawn CNN, and soft-voting ensemble weights are
> available **upon request for verified research collaboration**.
>
> Contact: `shrutiagarwaljsr@gmail.com` with your institutional affiliation
> and intended research use.

The Gradio UI runs in **graceful demo mode** automatically when weights are absent.

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Baseline CNN | 0.5000 | 0.9131 |
| CBAM-CNN | 0.9862 | 0.9989 |
| MobileNetV2 | 0.9862 | 0.9862 |
| **Ensemble** | **0.9954** | **0.9998** |


---

## 📖 Citation
If you use this work, please cite:
\```bibtex
@misc{drowsyguard2026,
  title  = {DrowsyGuard: Real-Time Multi-Signal Driver Drowsiness Detection},
  author = {Shruti Agarwal},
  year   = {2026},
  note   = {GitHub: https://github.com/shrutiiagarwall/drowsiness-detection}
}
\```

## 📜 License

MIT License. Novel feature engineering methodology subject to separate IP disclosure.
