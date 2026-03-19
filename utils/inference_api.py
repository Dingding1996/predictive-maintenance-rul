"""utils/inference_api.py

FastAPI inference server for the turbofan engine health classifier.

Loads the best FLAML AutoML model from the MLflow Model Registry and exposes:

  POST /predict_file  — upload a C-MAPSS .txt file; API takes last 30 cycles for a given unit
  GET  /health        — liveness probe + model metadata

Run locally::

    uvicorn utils.inference_api:app --host 0.0.0.0 --port 8000 --reload

Or::

    python utils/inference_api.py

Docker (recommended)::

    docker compose up --build
    # Swagger UI → http://localhost:8001/docs
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

# Ensure project root is importable whether invoked directly or via uvicorn
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ml_classification import CLASS_NAMES  # noqa: E402
from utils.data_loader import USEFUL_SENSORS, OP_SETTINGS, COLUMN_NAMES  # noqa: E402
from utils.feature_engineering import (  # noqa: E402
    add_rolling_features_spark,
    add_cycle_normalisation_spark,
    apply_min_max_spark,
    get_feature_cols,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE_DIR  = Path(__file__).parent.parent
MLRUNS_URI = f"file:///{_BASE_DIR / 'mlruns'}"
MODEL_NAME = "turbofan-health-classification"


# ---------------------------------------------------------------------------
# Model loading — robust path normalisation for Docker mounts
# ---------------------------------------------------------------------------
# The registry records Windows absolute paths (file:///C:/...) on the host.
# Inside the container mlruns/ is mounted at /app/mlruns, so we resolve the
# artifact path relative to mlruns/ so the same registry works on any OS.

def _load_registered(name: str) -> object:
    """Load the latest version of a registered sklearn model from MLflow.

    Reads the artifact path from meta.yaml directly — bypasses the MLflow
    client API to avoid Windows→Linux path issues when running in Docker.

    Args:
        name: Registered model name in the MLflow Model Registry.

    Returns:
        Loaded sklearn-compatible model (LightGBM, XGBoost, etc.).

    Raises:
        RuntimeError: If no registered versions are found.
    """
    models_dir   = _BASE_DIR / "mlruns" / "models" / name
    version_dirs = sorted(
        models_dir.glob("version-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not version_dirs:
        raise RuntimeError(f"No versions found for registered model '{name}'")

    meta = yaml.safe_load((version_dirs[-1] / "meta.yaml").read_text())

    # MLflow 2.x uses 'source'; 3.x uses 'storage_location' — handle both
    raw_path: str = meta.get("source") or meta.get("storage_location", "")

    # Normalise: strip file:/// prefix and backslashes
    normalised = raw_path.replace("\\", "/")
    for prefix in ("file:///", "file://"):
        if normalised.startswith(prefix):
            normalised = normalised[len(prefix):]

    # Resolve relative to the local mlruns/ mount point
    marker = "mlruns/"
    idx    = normalised.find(marker)
    if idx == -1:
        raise RuntimeError(f"Cannot locate 'mlruns/' in artifact path: {raw_path}")
    relative      = normalised[idx + len(marker):]
    artifact_path = _BASE_DIR / "mlruns" / relative

    return mlflow.sklearn.load_model(str(artifact_path))


# ---------------------------------------------------------------------------
# Global model registry — populated during lifespan startup
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load models once before the server starts handling requests."""
    print("Starting Predictive Maintenance Inference API…")
    mlflow.set_tracking_uri(MLRUNS_URI)
    try:
        _REGISTRY["automl"] = _load_registered(MODEL_NAME)
        client   = mlflow.MlflowClient()
        versions = client.get_registered_model(MODEL_NAME).latest_versions
        _REGISTRY["run_id"] = versions[-1].run_id if versions else "unknown"
        estimator = type(_REGISTRY["automl"]).__name__
        print(f"  [automl]  models:/{MODEL_NAME}/latest  ({estimator})  run={_REGISTRY['run_id'][:8]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [automl]  WARNING — model failed to load: {exc}")

    # Load min-max scale_params for feature normalisation
    scale_path = _BASE_DIR / "models" / "scale_params.json"
    if scale_path.exists():
        raw = json.loads(scale_path.read_text())
        _REGISTRY["scale_params"] = {k: tuple(v) for k, v in raw.items()}
        print(f"  [scale]   scale_params loaded ({len(_REGISTRY['scale_params'])} cols)")
    else:
        print("  [scale]   WARNING — models/scale_params.json not found. "
              "Re-run Section 5 of the notebook with scale_params=scale_params.")
    yield
    print("Shutting down — clearing model registry.")
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Turbofan Engine Health Classification API",
    description=(
        "3-class health classification for NASA C-MAPSS turbofan engines.\n\n"
        "**Classes:** `0 = Healthy` | `1 = Degrading` | `2 = Critical`\n\n"
        "**Model:** Best estimator selected by FLAML AutoML, registered in MLflow.\n\n"
        "Upload a C-MAPSS test file to `POST /predict_file` — the full feature "
        "engineering pipeline runs server-side."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    """Prediction result for one engine unit."""

    predictions:   list[int]         = Field(..., description="Predicted class index.")
    labels:        list[str]         = Field(..., description="Human-readable health label.")
    probabilities: list[list[float]] = Field(..., description="Softmax probabilities [Healthy, Degrading, Critical].")
    estimator:     str               = Field(..., description="FLAML-selected model family (e.g. 'LGBMClassifier').")
    run_id:        str               = Field(..., description="MLflow run ID (first 8 chars) for audit trail.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health_check() -> dict:
    """Liveness probe — returns service status and loaded model metadata."""
    loaded = "automl" in _REGISTRY
    run_id = _REGISTRY.get("run_id", "not loaded")
    return {
        "status":      "ok" if loaded else "degraded",
        "model":       MODEL_NAME,
        "run_id":      run_id[:8] if len(run_id) > 8 else run_id,
        "class_names": CLASS_NAMES,
    }


@app.post("/predict_file", response_model=PredictResponse, tags=["Inference"])
async def predict_file(
    file: UploadFile = File(..., description="C-MAPSS test .txt file (space-delimited, no header)."),
    unit_id: int = Query(1, description="Engine unit ID to predict (must exist in the file)."),
    window: int = Query(30, ge=1, description="Number of most recent cycles to use for rolling features."),
) -> PredictResponse:
    """
    Classify engine health from an uploaded C-MAPSS test file.

    Upload any `test_FD00X.txt` file from the NASA C-MAPSS dataset.
    The API automatically selects the last `window` cycles for the specified
    engine unit, runs the full feature engineering pipeline server-side,
    and returns a health prediction.

    Args:
        file:    C-MAPSS .txt file — space-delimited, no header, 26 columns
                 (unit_nr, time_cycles, 3 op_settings, 21 sensors).
        unit_id: Engine unit number to evaluate (default: 1).
        window:  Rolling window size in cycles (default: 30, matches training).

    Returns:
        Predicted health label, class index, probabilities, and run ID.
    """
    model        = _REGISTRY.get("automl")
    scale_params = _REGISTRY.get("scale_params")

    if model is None:
        raise HTTPException(status_code=503, detail="AutoML model is not loaded.")
    if scale_params is None:
        raise HTTPException(
            status_code=503,
            detail="scale_params not found. Re-run Section 5 with scale_params=scale_params.",
        )
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="File must be a .txt file (C-MAPSS format).")

    try:
        content = await file.read()
        df = pd.read_csv(
            io.StringIO(content.decode("utf-8")),
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc

    if unit_id not in df["unit_nr"].values:
        available = sorted(df["unit_nr"].unique().tolist())
        raise HTTPException(
            status_code=400,
            detail=f"unit_id {unit_id} not found in file. Available units: {available}",
        )

    # Take the last `window` cycles — more cycles give rolling stats that better
    # match the training distribution
    unit_df = (
        df[df["unit_nr"] == unit_id]
        .sort_values("time_cycles")
        .tail(window)
        .reset_index(drop=True)
    )
    unit_df["time_cycles"] = range(1, len(unit_df) + 1)

    try:
        unit_df      = add_rolling_features_spark(unit_df, sensor_cols=USEFUL_SENSORS, window_size=window)
        unit_df      = add_cycle_normalisation_spark(unit_df)
        # include_norm_cycle=False matches training (notebook Section 3b)
        feature_cols = get_feature_cols(USEFUL_SENSORS, op_settings=OP_SETTINGS, include_norm_cycle=False)
        unit_df      = apply_min_max_spark(unit_df, scale_params, feature_cols)

        X      = unit_df[feature_cols].iloc[[-1]].values.astype(np.float32)
        y_pred = model.predict(X)
        probas = model.predict_proba(X)
        labels = [CLASS_NAMES[i] for i in y_pred]

        return PredictResponse(
            predictions   = y_pred.tolist(),
            labels        = labels,
            probabilities = probas.tolist(),
            estimator     = model.__class__.__name__,
            run_id        = _REGISTRY.get("run_id", "unknown")[:8],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("utils.inference_api:app", host="0.0.0.0", port=8000, reload=False)
