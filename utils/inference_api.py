"""utils/inference_api.py

FastAPI inference server for the turbofan engine health classifier.

Supports two model endpoints:
  POST /predict/xgboost  — single-row tabular feature vector (rolling-window features)
  POST /predict/lstm     — sequence of raw sensor readings (seq_len × n_features)
  GET  /health           — liveness probe

Environment variables (override defaults):
  XGB_MODEL_PATH   : path to the saved XGBoost .pkl file
  LSTM_MODEL_PATH  : path to the saved LSTM .keras file

Run locally::

    uvicorn utils.inference_api:app --host 0.0.0.0 --port 8000 --reload

Or from the project root::

    python -m uvicorn utils.inference_api:app --port 8000
"""

from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.ml_classification import CLASS_NAMES, load_model

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

XGB_MODEL_PATH: str  = os.getenv("XGB_MODEL_PATH",  "models/xgb_pipeline.pkl")
LSTM_MODEL_PATH: str = os.getenv("LSTM_MODEL_PATH", "models/lstm_model.keras")


# ---------------------------------------------------------------------------
# Model registry — loaded once at startup
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Any] = {}


def _load_registry() -> None:
    """Attempt to load both models into the in-memory registry at startup."""
    for model_type, path_str in [("xgboost", XGB_MODEL_PATH), ("lstm", LSTM_MODEL_PATH)]:
        path = Path(path_str)
        if path.exists():
            try:
                _REGISTRY[model_type] = load_model(path, model_type=model_type)
                print(f"  [registry] {model_type} model loaded from {path}")
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Failed to load {model_type} model: {exc}", RuntimeWarning)
        else:
            warnings.warn(
                f"  [registry] {model_type} model not found at {path} — "
                "endpoint will return 503.",
                RuntimeWarning,
            )


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load models before the server starts serving requests."""
    print("Starting Predictive Maintenance Inference API…")
    _load_registry()
    yield
    print("Shutting down — clearing model registry.")
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance — Engine Health Classifier",
    description=(
        "Classifies turbofan engine health as **Healthy / Degrading / Critical** "
        "using either an XGBoost (tabular) or LSTM (sequence) model trained on "
        "the NASA C-MAPSS dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class XGBRequest(BaseModel):
    """Single-row tabular feature vector for the XGBoost endpoint."""

    features: list[float] = Field(
        ...,
        description=(
            "Flat list of rolling-window features in the same order as the "
            "training columns (op_settings + norm_cycle + sensor rolling stats). "
            "Length must match the training feature dimension."
        ),
        min_length=1,
    )

    model_config = {"json_schema_extra": {"example": {"features": [0.5] * 73}}}


class LSTMRequest(BaseModel):
    """Sliding-window sequence of sensor readings for the LSTM endpoint."""

    sequence: list[list[float]] = Field(
        ...,
        description=(
            "2-D array of shape (seq_len, n_features) representing a "
            "contiguous window of sensor readings.  seq_len and n_features "
            "must match the model's expected input shape."
        ),
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {"sequence": [[0.5] * 14 for _ in range(30)]}
        }
    }


class PredictionResponse(BaseModel):
    """Standard response for both prediction endpoints."""

    health_class: int = Field(..., description="Predicted class index (0 / 1 / 2).")
    health_label: str = Field(..., description="Human-readable label (Healthy / Degrading / Critical).")
    confidence:   float = Field(..., ge=0.0, le=1.0, description="Softmax probability of the predicted class.")
    probabilities: list[float] = Field(..., description="Softmax probabilities for all classes [Healthy, Degrading, Critical].")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_response(probabilities: np.ndarray) -> PredictionResponse:
    """Convert a softmax probability vector into a PredictionResponse.

    Args:
        probabilities: 1-D array of shape (n_classes,).

    Returns:
        PredictionResponse with class index, label, confidence, and full probs.
    """
    idx = int(np.argmax(probabilities))
    return PredictionResponse(
        health_class=idx,
        health_label=CLASS_NAMES[idx],
        confidence=float(probabilities[idx]),
        probabilities=[float(p) for p in probabilities],
    )


def _require_model(model_type: str) -> Any:
    """Fetch a model from the registry or raise 503 if not available.

    Args:
        model_type: 'xgboost' or 'lstm'.

    Returns:
        Loaded model.

    Raises:
        HTTPException 503: If the model was not loaded at startup.
    """
    model = _REGISTRY.get(model_type)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{model_type} model is not available.  "
                f"Ensure the model file exists at the configured path and "
                f"restart the server."
            ),
        )
    return model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health_check() -> dict[str, str]:
    """Liveness probe — returns service status and loaded models.

    Returns:
        Dict with 'status' and 'models_loaded' keys.
    """
    loaded = [k for k in ["xgboost", "lstm"] if k in _REGISTRY]
    return {
        "status": "ok",
        "models_loaded": ", ".join(loaded) if loaded else "none",
    }


@app.post(
    "/predict/xgboost",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="XGBoost health prediction from rolling-window features",
)
def predict_xgboost(request: XGBRequest) -> PredictionResponse:
    """Predict engine health class from a tabular rolling-window feature vector.

    The feature vector must match the column order used during training.
    Consult `utils.feature_engineering.get_feature_cols` for the expected order.

    Args:
        request: XGBRequest containing the flat feature vector.

    Returns:
        PredictionResponse with class, label, confidence, and all probabilities.
    """
    pipeline = _require_model("xgboost")

    X = np.array(request.features, dtype=np.float32).reshape(1, -1)

    try:
        # XGBClassifier inside Pipeline does not expose predict_proba by default
        # when objective='multi:softmax'; use predict_proba if available
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)[0]
        else:
            # Fallback: one-hot encode predicted class
            pred_class = int(pipeline.predict(X)[0])
            probs = np.zeros(len(CLASS_NAMES))
            probs[pred_class] = 1.0
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return _build_response(probs)


@app.post(
    "/predict/lstm",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="LSTM health prediction from sensor time-series sequence",
)
def predict_lstm(request: LSTMRequest) -> PredictionResponse:
    """Predict engine health class from a sliding-window sensor sequence.

    The sequence shape must match (seq_len, n_features) used at training time.
    Use `utils.feature_engineering.USEFUL_SENSORS` to determine expected columns.

    Args:
        request: LSTMRequest containing the 2-D sensor sequence.

    Returns:
        PredictionResponse with class, label, confidence, and all probabilities.
    """
    model = _require_model("lstm")

    try:
        X = np.array(request.sequence, dtype=np.float32)[np.newaxis, ...]  # (1, T, F)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse sequence into a numeric array: {exc}",
        ) from exc

    expected_shape = model.input_shape  # (None, seq_len, n_features)
    if expected_shape[1] is not None and X.shape[1] != expected_shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Sequence length mismatch: expected {expected_shape[1]}, "
                f"got {X.shape[1]}."
            ),
        )
    if expected_shape[2] is not None and X.shape[2] != expected_shape[2]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature dimension mismatch: expected {expected_shape[2]}, "
                f"got {X.shape[2]}."
            ),
        )

    try:
        probs = model.predict(X, verbose=0)[0]  # shape (n_classes,)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"LSTM inference failed: {exc}") from exc

    return _build_response(probs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "utils.inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
