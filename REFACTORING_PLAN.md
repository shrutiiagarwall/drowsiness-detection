# Refactoring Plan: Monolithic Notebook → Production Repository

This document explains exactly how to split `Fixed_Drowsiness_Detection.ipynb`
into the production repository structure, with special attention to **abstracting
the novel feature engineering logic** without breaking the pipeline.

---

## 1. The Core Abstraction Strategy (IP Protection)

The proprietary logic lives in **`src/features/feature_extractor.py`**.
The strategy is a **protocol-conformant stub**:

```
┌────────────────────────────────────────┐
│  Public repo                           │
│  OcularFeatureExtractor (stub)         │  ← This file is in git
│  • Real API / method signatures        │
│  • Docstrings explaining the "what"    │
│  • Stub logic (gradient magnitude,     │
│    basic statistics)                   │
│  • Identical input/output contracts    │
└─────────────────┬──────────────────────┘
                  │  drop-in replacement
┌─────────────────▼──────────────────────┐
│  Private / on-request                  │
│  _RealOcularFeatureExtractor           │  ← NOT in git
│  • Proprietary multi-scale transform   │
│  • Calibrated thresholds               │
│  • Optimised kernel parameters         │
└────────────────────────────────────────┘
```

**Key principle:** All downstream code (models, inference engine, Gradio UI)
consumes only `OcularFeatureVector` outputs. They never import the internals.
Swapping in the real implementation is a one-line change in
`app/inference_engine.py` and `scripts/train.py`.

---

## 2. Notebook → File Mapping

### Notebook Section 1 (Environment Setup)
→ **`requirements.txt`** and **`configs/config.yaml`**

All `!pip install` lines become `requirements.txt`.
All magic numbers (`IMG_SIZE = 48`, `BATCH_SIZE = 64`, `PERCLOS_THRESH = 0.7`)
move to `configs/config.yaml`. Scripts load config via `yaml.safe_load()`.

---

### Notebook Sections 2A–2D (Dataset Loading, CLAHE, Visualisation)
→ **`src/preprocessing/loader.py`**
→ **`notebooks/01_eda_and_preprocessing.ipynb`** (trimmed)

- `apply_clahe()` → `src/preprocessing/loader.py:apply_clahe()`
- `load_dataset()` → `src/preprocessing/loader.py:load_dataset()`
- `dataset_inventory()` → `src/preprocessing/loader.py:dataset_inventory()`
- Visualisation code (`plot_class_distribution`, `show_sample_grid`)
  → `src/utils/visualisation.py` (keep in notebook for EDA story)

The EDA notebook keeps the *visualisation cells* to tell the data story
for recruiters and reviewers. The *logic* is imported from `src/`.

---

### Notebook Section 3 (Augmentation, SMOTE, TF Tensors)
→ **`src/preprocessing/augmentation.py`**
→ **`src/preprocessing/tensors.py`**

- `augment_pipeline` / `augment_dataset()` → `augmentation.py`
- `apply_smote()` → `augmentation.py`
- `prepare_tensors()` + `make_dataset()` → `tensors.py`

---

### Notebook Section 4 (CBAM Architecture)
→ **`src/models/cbam_cnn.py`**

The full CBAM architecture is **public** — it is a standard academic
building block (Woo et al. 2018). Move `ReduceMeanLayer`, `ReduceMaxLayer`,
`channel_attention()`, `spatial_attention()`, `cbam_block()`, and
`build_cbam_cnn()` verbatim.

---

### Notebook Section 5 (Transfer Models + Ensemble)
→ **`src/models/transfer_models.py`** (add MobileNetV2 / EfficientNetB0 wrappers)
→ **`src/models/ensemble.py`** (`SoftVotingEnsemble`)

---

### Notebook Section 6 (Training Loop & Callbacks)
→ **`scripts/train.py`**

The training loop is fully public. Move it to a CLI script that reads
from `configs/config.yaml`. Keep a **stripped-output copy** in
`notebooks/02_model_training.ipynb` for narrative context.

---

### Notebook Section 7 (Grad-CAM)
→ **`src/utils/gradcam.py`**
→ **`notebooks/03_evaluation_and_xai.ipynb`**

---

### Notebook Section 8 (Evaluation Metrics + Dashboard)
→ **`notebooks/03_evaluation_and_xai.ipynb`**
→ **`src/utils/visualisation.py`**

---

### Novel Feature Engineering (the part to HIDE)
Your notebook likely contains in Sections 9–10 (Inference Engine /
`FusionDrowsinessScorer`) some calibration constants or threshold-
derivation logic that is novel.

**Action:**
1. Move the *interface* (`FusionDrowsinessScorer`) to `src/features/perclos.py` — **public**.
2. In the notebook cells that contained the derivation maths, replace them with:

```python
# ============================================================
# SECTION: Fusion Weight Calibration
# ============================================================
# The empirical calibration procedure that derives the fusion
# weights (eye_weight, yawn_weight) and decision thresholds
# (alert_threshold, warning_threshold) is described in
# Section 4.3 of the accompanying paper.
#
# The calibrated values are loaded from configs/config.yaml.
# The derivation code is available under research collaboration
# agreement — contact: [your-email@institution.edu]
# ============================================================
import yaml
with open('../configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

inference_cfg = cfg['inference']
print("Fusion config loaded:", inference_cfg)
```

This lets the notebook *run* while hiding the derivation.

---

### Notebook Section 9 (Drift Detection)
→ **`src/utils/drift_detector.py`**
→ **`notebooks/04_drift_detection.ipynb`**

---

### Notebook Section 11 (Dashboard)
→ **`notebooks/03_evaluation_and_xai.ipynb`** (keep as-is for storytelling)
→ **`src/utils/visualisation.py`** (extract plot functions)

---

### Notebook Section 12 (FastAPI)
→ **`app/api.py`** (production-ready version provided)

---

## 3. Git Hygiene Checklist

Before pushing:

```bash
# 1. Strip all notebook outputs (prevent accidental data leaks)
pip install nbstripout
nbstripout notebooks/*.ipynb

# 2. Verify no weights are staged
git status | grep -E "\.h5|\.keras|\.pkl|\.pt"
# Should print nothing.

# 3. Verify .gitignore is respected
git ls-files --others --ignored --exclude-standard | head -20

# 4. Set up pre-commit hook to auto-strip notebooks
echo '#!/bin/sh\nnbstripout --install' >> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## 4. HuggingFace Spaces Deployment

The root `app.py` is the Spaces entry point. For Spaces to work:

1. Create `README.md` frontmatter:

```yaml
---
title: DrowsyGuard
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: true
license: mit
---
```

2. The app runs in **DEMO mode** on Spaces (no weights), which is intentional —
   it demonstrates the full pipeline architecture without exposing weights.
