# Predictive Maintenance — Turbofan Engine Health Classification

Multi-class health classification on the NASA C-MAPSS turbofan engine dataset,
demonstrating an end-to-end MLOps pipeline from raw sensor data to a containerised
inference API with automated model selection and experiment tracking.

---

## Problem Statement

Predict the current health state of a jet engine from its sensor history, enabling
maintenance teams to intervene before catastrophic failure.

**Health classes** (derived from Remaining Useful Life):

| Class | RUL range | Meaning |
|---|---|---|
| Healthy (0) | > 80 cycles | Normal operation |
| Degrading (1) | 31 – 80 cycles | Early-warning window |
| Critical (2) | ≤ 30 cycles | Urgent maintenance required |

---

## Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
Source: [Kaggle — behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

| Sub-dataset | Operating conditions | Fault modes | Train engines | Test engines |
|---|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 | 100 |
| FD002 | 6 | HPC degradation | 260 | 259 |
| FD003 | 1 | HPC + fan degradation | 100 | 100 |
| FD004 | 6 | HPC + fan degradation | 249 | 248 |

The dataset ships with a **predefined engine-level train/test split**:
- `train_FD00X.txt` — complete run-to-failure trajectories; labels derived internally
- `test_FD00X.txt` — truncated sequences of entirely unseen engines
- `RUL_FD00X.txt` — ground-truth RUL at the last observed test cycle (used to reconstruct `y_test`)

Dataset is downloaded automatically via `kagglehub` on first run.

---

## Project Structure

```
predictive-maintenance-rul/
├── PredictiveMaintenance_Training.ipynb   # Main notebook (full pipeline)
├── requirements.txt                       # Full training dependencies
├── requirements-inference.txt             # Minimal inference-only dependencies
├── Dockerfile                             # Container for inference API
├── docker-compose.yml                     # Compose config (port 8001)
├── assets/                                # Exported figures
├── data/
│   └── raw/                               # Local data cache (gitignored — auto-downloaded)
├── mlruns/                                # MLflow experiment tracking & model registry
├── models/                                # Saved artefacts (gitignored — regenerate via notebook)
│   ├── scale_params.json                  # Min-max scale parameters for raw-data inference
│   └── lstm_model.keras                   # Optional LSTM checkpoint
└── utils/
    ├── data_loader.py                     # C-MAPSS loader + RUL computation
    ├── download_dataset.py                # kagglehub download helper
    ├── feature_engineering.py             # Rolling-window feature engineering
    ├── ml_classification.py               # FLAML AutoML + LSTM wrappers, evaluation
    ├── inference_api.py                   # FastAPI inference service
    └── plot_style.py                      # Global plot theme and colour palettes
```

---

## Pipeline Overview

```
Data Acquisition → EDA → Preprocessing → AutoML → Evaluation → Explainability → Deployment
```

| Section | What happens |
|---|---|
| 1. Data Acquisition | Download C-MAPSS via kagglehub; load train + test splits for FD001–FD004 |
| 2. EDA | Class distribution, sensor degradation trends, correlation heatmap |
| 3. Preprocessing | RUL labelling → data-driven threshold search → operating-condition normalisation → 30-cycle rolling features → 100-cycle long-window features (lw_mean, lw_std, slope) → min-max scaling |
| 4. Model Definition | FLAML AutoML configuration; optional LSTM sequence model |
| 5. AutoML & Model Selection | FLAML searches LightGBM / XGBoost / RF / linear models; best run logged to MLflow |
| 6. Evaluation | Test-set metrics (accuracy, F1-macro, F1-weighted) on unseen engines |
| 7. Visualisations | Confusion matrices, feature importance, LSTM training curves |
| 8. Explainability | SHAP TreeExplainer — beeswarm + global bar chart |
| 9. Summary | Key findings and limitations |

---

## Models

### FLAML AutoML (primary)
- Searches over LightGBM, XGBoost, Random Forest, Extra Trees, and linear models
- Optimises F1-weighted using a cost-frugal Bayesian search strategy
- Best model and hyperparameters registered automatically in MLflow Model Registry
- Input: flat feature vector (115 features per cycle: 14 sensors × 8 statistics + 3 op_settings)

### LSTM (optional — requires TensorFlow)
- Input: sequences of engineered features `(30 cycles × 115 features)` — same feature set as AutoML
- 2-layer LSTM (128 → 64 units) with dropout, early stopping, and class-weighted training
- Captures temporal degradation patterns across consecutive cycles end-to-end

---

## Key Results

> Results below are from the current clean pipeline: official engine-level train/test split,
> operating-condition normalisation, and leakage-free scaling. Re-run the notebook to reproduce.

| Model | CV F1-weighted | Test F1-weighted | Test F1-macro | Test Accuracy |
|---|---|---|---|---|
| **FLAML AutoML (LightGBM)** | — | — | — | — |
| LSTM | — | — | — | — |

*Numbers to be updated after notebook re-run with the corrected pipeline.*

**Previous results (row-level split, no condition normalisation — inflated by leakage):**

| Model | Test F1-weighted | Test Accuracy |
|---|---|---|
| FLAML AutoML (LightGBM) | 0.9877 | 0.9877 |

Those numbers should not be compared against the current pipeline — the evaluation setup was fundamentally different (test cycles from the same engines seen in training).

---

## Explainability

SHAP values identify the most predictive features for the **Critical** class
(update after re-run — list below reflects previous pipeline):

- `sensor_14_mean` / `sensor_14_std` — core speed tracks late-stage degradation
- `sensor_11_mean` — HPC static pressure rises with compressor fouling
- `sensor_04_delta` — rate-of-change in LPT outlet temperature

Note: `norm_cycle` is **excluded** from features — it encodes lifecycle position derived
from the failure point and constitutes target leakage.

![SHAP global importance](assets/shap_global_importance.png)

---

## MLflow Experiment Tracking

All training runs are logged to the local MLflow Model Registry under the
experiment `turbofan-health-classification`.

```bash
# Launch the MLflow UI from the project root
python -m mlflow ui
# Open http://127.0.0.1:5000
```

Each run records:
- All Section 0 constants (window size, RUL thresholds, number of op-condition clusters, etc.)
- FLAML search configuration (time budget, metric)
- Best estimator name and hyperparameters
- CV F1-weighted, test accuracy, test F1-macro, test F1-weighted
- Trained model artifact with input/output schema

---

## Inference API

### Local (no Docker)

```bash
uvicorn utils.inference_api:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### Docker (recommended)

```bash
docker compose up --build
# Swagger UI → http://localhost:8001/docs
```

### Endpoints

| Method | Endpoint | Input | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness probe + model metadata |
| POST | `/predict_file` | C-MAPSS `.txt` file upload | Full pipeline server-side: condition normalisation → rolling features → min-max → predict |

**`/predict_file` example** — upload a raw C-MAPSS test file, get health label back:

```bash
curl -X POST "http://localhost:8001/predict_file?unit_id=1" \
  -F "file=@data/raw/test_FD001.txt"
# → {"labels": ["Healthy"], "predictions": [0], "probabilities": [[0.91, 0.07, 0.02]], ...}
```

The `mlruns/` and `models/` directories are **mounted as read-only volumes** — not
baked into the image. Retraining on the host and running `docker compose restart`
is all that is needed to deploy a new model version.

---

## CI/CD Design

The training pipeline is designed for CI/CD integration via GitHub Actions.
On each merge to `main`, the notebook can be executed automatically, the new
model version registered to MLflow, and the container restarted to serve the
updated model — decoupling training from deployment.

```
git push → GitHub Actions → jupyter nbconvert --execute → MLflow register → docker compose restart
```

---

## Setup

**Requirements:** Python 3.11, pip

```bash
pip install -r requirements.txt
```

**Kaggle API token** — required for automatic dataset download:

```bash
# Place kaggle.json in your home directory (Kaggle → Settings → API → New Token)
# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle
Copy-Item kaggle.json $env:USERPROFILE\.kaggle\
```

**Run the notebook:**

```bash
jupyter lab PredictiveMaintenance_Training.ipynb
```

---

## Limitations & Next Steps

- Health-class thresholds are searched data-driven (ExtraTrees proxy grid search) but remain a proxy for true maintenance economics; a cost-sensitive objective could further improve Critical recall
- Transformer-based sequence models (e.g. Temporal Fusion Transformer) may outperform
  LSTM with less hyperparameter tuning
- The k-means operating-condition clustering uses k=6 (matching FD002/FD004's known
  regimes); a data-driven k selection (elbow / silhouette) would be more principled
- A shared MLflow tracking server would unify experiment history across multiple
  portfolio projects
