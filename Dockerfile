# =============================================================
# Predictive Maintenance — Inference Service
# =============================================================
# Serves the FLAML AutoML model as a REST API via FastAPI.
# The model artefacts (mlruns/) are mounted at runtime — not
# baked into the image — so the same image works with any
# trained checkpoint.
#
# Build:
#   docker build -t turbofan-health-api .
#
# Run (standalone):
#   docker run -p 8001:8000 \
#     -v "$(pwd)/mlruns:/app/mlruns:ro" \
#     turbofan-health-api
#
# Run via Compose (recommended):
#   docker compose up
# =============================================================

FROM python:3.11-slim

# --- System dependencies ---
# gcc required to compile C extensions (e.g. lightgbm wheels fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
# Copy requirements first to leverage Docker layer caching.
# Package reinstall only happens when this file changes.
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# --- Application code ---
# Only the utils/ modules needed for inference — notebooks, data,
# and training artefacts remain on the host.
COPY utils/ ./utils/

# --- Runtime ---
EXPOSE 8000

# mlruns/ and models/ are NOT copied — they must be mounted as volumes at runtime.
# This keeps the image model-agnostic: re-training on the host and restarting
# the container is all that's needed to deploy a new model version.

CMD ["uvicorn", "utils.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
