"""
app/api.py
===========
FastAPI REST API — production deployment endpoint.

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference_engine import (
    load_model_or_mock,
    preprocess_frame,
    run_inference,
    MockModel,
)

import cv2

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DrowsyGuard API",
    description=(
        "Driver drowsiness detection via CBAM-CNN ensemble. "
        "Runs in demo mode when weights are absent."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = ROOT / "models" / "weights" / "cbam_cnn_final.keras"
model = load_model_or_mock(WEIGHTS_PATH)

_metrics: dict = {
    "total": 0,
    "closed": 0,
    "open": 0,
    "start_time": time.time(),
}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    image_b64: str = Field(
        ..., description="Base64-encoded JPEG/PNG of the cropped eye region."
    )
    apply_clahe: bool = Field(True, description="Apply CLAHE before inference.")


class PredictResponse(BaseModel):
    state: str
    label: int
    confidence: float
    is_drowsy: bool
    latency_ms: float
    demo_mode: bool


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    demo_mode: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    closed_count: int
    open_count: int
    drowsy_rate: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _metrics["start_time"], 1),
        demo_mode=isinstance(model, MockModel),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def get_metrics():
    """Prediction counts since server start."""
    total = _metrics["total"]
    return MetricsResponse(
        total_predictions=total,
        closed_count=_metrics["closed"],
        open_count=_metrics["open"],
        drowsy_rate=_metrics["closed"] / max(total, 1),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Classify a single eye-region image.

    Returns the predicted state (Closed/Open), confidence, drowsiness flag,
    and inference latency.
    """
    t0 = time.perf_counter()
    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    tensor = preprocess_frame(bgr, img_size=48, apply_clahe=req.apply_clahe)
    result = run_inference(model, tensor)
    latency_ms = (time.perf_counter() - t0) * 1000

    _metrics["total"] += 1
    _metrics["closed" if result.is_drowsy else "open"] += 1

    return PredictResponse(
        state=result.label_name,
        label=result.label,
        confidence=round(result.confidence, 4),
        is_drowsy=result.is_drowsy,
        latency_ms=round(latency_ms, 2),
        demo_mode=result.is_mock,
    )
