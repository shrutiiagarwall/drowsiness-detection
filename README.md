# 🚗 DrowsyGuard: Real-Time Multi-Signal Driver Drowsiness Detection

<p align="center">
  <img src="assets/banner.png" alt="DrowsyGuard Banner" width="800"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow&logoColor=white" alt="TensorFlow"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Gradio-4.x-purple?logo=gradio&logoColor=white" alt="Gradio"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" alt="Status"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Pending%20Submission-red" alt="Paper"/></a>
</p>

---

## 📄 Abstract

> *[Placeholder — to be updated upon journal submission.]*
>
> Driver drowsiness is a major contributor to road fatalities worldwide. Existing detection systems typically rely on a single signal (eye state or yawning) and lack clinical interpretability. This work presents **DrowsyGuard**, a multi-signal ensemble framework that fuses eye-state classification (PERCLOS metric) with yawn detection via a weighted temporal fusion score. The architecture introduces a **Convolutional Block Attention Module (CBAM)** on top of a CNN backbone, enabling the model to attend to diagnostically salient regions. We further integrate **Grad-CAM explainability**, **ADWIN-based concept drift detection**, and a **TFLite INT8 export** for edge deployment. Experiments on a balanced dataset of ocular and oral regions demonstrate improvements over single-signal baselines across accuracy, ROC-AUC, and clinical PERCLOS alignment.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│     Webcam Frame / Uploaded Image / REST Payload                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Face & Region Detection  │
              │  (Haar Cascade / MediaPipe)│
              └──────┬────────────┬───────┘
                     │            │
          ┌──────────▼──┐   ┌─────▼──────────┐
          │  Eye Region │   │  Mouth Region  │
          │  Extractor  │   │  Extractor     │
          └──────┬──────┘   └──────┬─────────┘
                 │                 │
   ┌─────────────▼──────┐  ┌──────▼──────────────┐
   │  CBAM–CNN Ensemble │  │  Yawn CNN Classifier │
   │  (Eye State Model) │  │  (Mouth State Model) │
   │  Baseline CNN      │  │                      │
   │  + MobileNetV2     │  │                      │
   │  + EfficientNetB0  │  │                      │
   └─────────────┬──────┘  └──────┬───────────────┘
                 │                │
         ┌───────▼────────────────▼──────┐
         │     Temporal Fusion Scorer    │
         │  PERCLOS × 0.6 + Yawn × 0.4  │
         │  Sliding window (90 frames)   │
         └───────────────┬───────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     Alert Decision Engine   │
          │  OK / WARNING / ALERT       │
          └──────────────┬──────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │        Output Layer                     │
    │  Gradio UI  |  FastAPI  |  TFLite Edge  │
    └─────────────────────────────────────────┘
```

### Key Research Contributions

| Research Gap | Solution |
|---|---|
| Black-box predictions | Grad-CAM explainability layer |
| Single-signal fragility | Eye + Yawn temporal fusion |
| No clinical metric | PERCLOS (% eye closure over time) |
| Static illumination assumption | CLAHE contrast normalization |
| Dataset imbalance | SMOTE + class-weighted loss |
| No production path | TFLite INT8 + FastAPI + Gradio |
| Concept drift in deployment | ADWIN statistical drift detector |
| Attention-blind convolution | CBAM (Channel + Spatial Attention) |

---

## 📁 Repository Structure

```
drowsiness-detection/
│
├── 📓 notebooks/
│   ├── 01_eda_and_preprocessing.ipynb     # Dataset analysis, CLAHE, visualisations
│   ├── 02_model_training.ipynb            # Architecture, ensemble, training loops
│   ├── 03_evaluation_and_xai.ipynb        # Metrics, Grad-CAM, PERCLOS timeline
│   └── 04_drift_detection.ipynb           # ADWIN concept drift experiments
│
├── 🔧 src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── loader.py                      # Dataset loading, CLAHE enhancement
│   │   ├── augmentation.py                # Albumentations pipeline
│   │   └── tensors.py                     # TF dataset builders
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py           # Public interface (stubs proprietary logic)
│   │   └── perclos.py                     # PERCLOS sliding-window scorer
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cbam_cnn.py                    # CBAM attention block + CNN backbone
│   │   ├── transfer_models.py             # MobileNetV2 / EfficientNetB0 wrappers
│   │   ├── ensemble.py                    # Soft-voting ensemble
│   │   └── tflite_exporter.py             # INT8 quantisation export
│   │
│   └── utils/
│       ├── __init__.py
│       ├── gradcam.py                     # Grad-CAM implementation
│       ├── drift_detector.py              # ADWIN wrapper
│       └── visualisation.py              # Plotting utilities
│
├── 🖥️  app/
│   ├── app.py                             # Gradio UI (main entry point)
│   ├── api.py                             # FastAPI REST server
│   └── inference_engine.py                # Real-time frame processing
│
├── 🗃️  data/
│   ├── raw/                               # ← git-ignored; user provides dataset
│   ├── processed/                         # ← git-ignored; generated by pipeline
│   └── samples/                           # Small demo images (tracked in git)
│
├── ⚙️  configs/
│   └── config.yaml                        # All hyper-parameters in one place
│
├── 🧪 tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_inference.py
│
├── 📜 scripts/
│   ├── train.py                           # CLI training script
│   └── export_tflite.py                   # TFLite export script
│
├── app.py                                 # ← Root-level alias (Gradio, for HuggingFace Spaces)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/drowsiness-detection.git
cd drowsiness-detection
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Prepare your dataset

The pipeline expects the following layout (standard eye-state dataset, e.g. MRL, CEW, or custom):

```
data/raw/
├── train/
│   ├── Closed/   *.jpg or *.png
│   └── Open/
└── test/
    ├── Closed/
    └── Open/
```

> **No dataset is bundled.** Place your images under `data/raw/` before running.

---

## 🚀 Quick Start

### Option A — Gradio Web UI

```bash
python app.py
```

Open `http://localhost:7860` in your browser. Upload a cropped eye image (or use your webcam) and receive a real-time drowsiness prediction with Grad-CAM overlay.

> **Note:** If model weights are not present, the UI runs in **demo mode** with mock predictions. See the section below.

### Option B — FastAPI REST Server

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

```bash
# Example inference call
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64-encoded-eye-image>", "apply_clahe": true}'
```

### Option C — Run Training Pipeline

```bash
python scripts/train.py --config configs/config.yaml
```

---

## 🤖 Model Weights & Pre-trained Artifacts

> **⚠️ Model weights and pre-trained artifacts are not included in this public repository.**
>
> The trained CBAM-CNN, MobileNetV2 fine-tuned weights, and the calibrated soft-voting ensemble are available **upon request for verified research collaboration**.
>
> To request access, please contact: `[your-email@institution.edu]` with:
> - Your institutional affiliation
> - A brief description of the intended research use
>
> Reviewers and collaborators associated with the pending publication will receive access automatically.

The Gradio UI and FastAPI server both include a **graceful mock mode** that activates automatically when weights are absent — so the full pipeline architecture is always demonstrable.

---

## 📊 Results

| Model | Accuracy | ROC-AUC | F1 (Drowsy) |
|---|---|---|---|
| Baseline CNN | — | — | — |
| CBAM-CNN | — | — | — |
| MobileNetV2 (fine-tuned) | — | — | — |
| **Ensemble (Soft-Vote)** | **—** | **—** | **—** |

> *Specific metric values are withheld pending peer review. They will be filled upon acceptance.*

---

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{drowsyguard2025,
  title   = {DrowsyGuard: Multi-Signal Ensemble Framework for Real-Time Driver Drowsiness Detection with Explainable AI},
  author  = {[Author Names]},
  journal = {[Journal / Conference — pending]},
  year    = {2025},
  note    = {Under review}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

The novel feature engineering methodology described in the accompanying paper is subject to a separate IP disclosure. The code in `src/features/feature_extractor.py` provides the public interface; the underlying proprietary computations are not included in this repository.
