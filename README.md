# Predictive Maintenance — Turbofan Engine Health Classification

Binary health classification on the NASA C-MAPSS turbofan engine dataset,
following the CRISP-DM methodology from raw sensor data to a containerised
inference API with automated model selection and experiment tracking.

---

## CRISP-DM Phases

### 1. Business Understanding

**Fault type:** High-Pressure Compressor (HPC) degradation and fan degradation in
jet turbofan engines — detectable via gradual shifts in temperature, pressure, and
speed sensor readings over hundreds of operational cycles.

**Problem framing:** Rather than regressing a precise RUL value, the task is
reframed as binary fault detection — a more actionable output for maintenance teams:

| Class | RUL range | Interpretation |
|---|---|---|
| Healthy (0) | > 80 cycles | Normal operation, no intervention required |
| Non-Healthy (1) | ≤ 80 cycles | Degradation detected — schedule maintenance |

The threshold (80 cycles) is searched data-driven via an ExtraTrees proxy grid-search
on training data, not assumed.

**Success metric:** **Recall on the Non-Healthy class** is the primary objective —
a missed fault leads to unplanned downtime, which is more costly than an unnecessary
maintenance check. F1 is the model selection metric; precision is monitored to keep
false alarm rates manageable.

---

### 2. Data Understanding

**Dataset:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
Source: [Kaggle — behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

| Sub-dataset | Operating conditions | Fault modes | Train engines | Test engines |
|---|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 | 100 |
| FD002 | 6 | HPC degradation | 260 | 259 |
| FD003 | 1 | HPC + fan degradation | 100 | 100 |
| FD004 | 6 | HPC + fan degradation | 249 | 248 |

**Key observations from EDA:**
- The dataset ships with a predefined **engine-level** train/test split — whole
  engines are either in train or test, with no cycle-level overlap.
- `train_FD00X.txt` contains complete run-to-failure trajectories; RUL is derived
  from the final observed cycle.
- `test_FD00X.txt` contains truncated sequences of unseen engines; `RUL_FD00X.txt`
  provides the ground-truth RUL at the last observed cycle.
- FD002 and FD004 operate under **6 distinct operating conditions**, creating
  multi-modal sensor distributions that must be handled before feature extraction.
- Class imbalance is present: early-lifecycle healthy cycles dominate, so optimising
  accuracy would mask poor fault detection.

Dataset is downloaded automatically via `kagglehub` on first run.

---

### 3. Data Preparation

The full preprocessing chain is implemented in [utils/feature_engineering.py](utils/feature_engineering.py)
and [utils/data_loader.py](utils/data_loader.py).

**Step 1 — RUL labelling and health class assignment**
RUL is computed as `max_cycle − current_cycle` per engine. Cycles with RUL ≤ 80
are labelled Non-Healthy (1); all others are Healthy (0).

**Step 2 — Operating-condition normalisation**
The same engine running at different loads produces different sensor amplitudes —
even when healthy. To remove this load-dependent bias before feature extraction:
- KMeans clustering (k=6) on the 3 operational settings identifies the operating
  regime of each cycle.
- Per-cluster z-score normalisation is applied to all 14 informative sensors:
  `z = (x − μ_condition) / σ_condition`
- Parameters are fit on training data only and applied to the test set.

**Step 3 — Rolling window feature extraction (short window, 30 cycles)**
For each of the 14 normalised sensors, a 30-cycle sliding window computes:
`mean, std, min, max, delta (last − first)`
This yields 70 features capturing recent degradation trends.

**Step 4 — Long-window trend features (60 cycles)**
A 60-cycle window extracts `lw_mean, lw_std, slope` per sensor (42 features),
capturing slower degradation trajectories that a short window would miss.

**Step 5 — Min-max scaling**
Final scaling is fit on training features only; scale parameters are saved to
`models/scale_params.json` and reused at inference time.

**Total engineered features:** 115 (3 op_settings + 14 sensors × 8 statistics + normalised cycle position)

---

### 4. Modelling

#### Model selection strategy

With labelled fault data available, supervised classification is appropriate.
Two approaches are evaluated:

**FLAML AutoML (primary)**
- Searches over LightGBM, XGBoost, Random Forest, Extra Trees, and linear models
- Directly optimises **F1 on the Non-Healthy class** — not accuracy
- **Engine-level GroupKFold CV** (5 folds): whole engines are held out per fold,
  eliminating same-engine cycle leakage and testing true generalisation to unseen assets
- Best model and hyperparameters registered automatically in MLflow Model Registry
- Decision threshold tuned post-training on a held-out validation fold

**LSTM (optional — requires TensorFlow)**
- Input: sequences of shape `(30 cycles × 115 features)`
- Architecture: LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(64, relu) → Dense(2, softmax)
- Class-weighted training to handle imbalance; early stopping (patience=10) and ReduceLROnPlateau

#### Temporal split discipline

The test set is touched **exactly once**: after FLAML completes its search and the
best model is selected from CV results alone. This mirrors the production scenario
where the model must generalise to engines it has never seen.

---

### 5. Evaluation

**Primary metrics** (accuracy is not reported as a standalone metric):

| Metric | Why it matters |
|---|---|
| **Recall (Non-Healthy)** | Measures missed fault rate — the most dangerous error |
| **F1 (Non-Healthy)** | Balances recall against false alarm rate |
| Precision (Non-Healthy) | Quantifies false alarm rate for operations teams |
| Confusion matrix | Translates statistical errors into business impact |

**Threshold tuning:** After FLAML selects the best model, a sweep over
`P(Non-Healthy) ∈ [0.10, 0.70]` identifies the threshold maximising F1 on a
held-out validation fold. The default (0.5) and tuned thresholds are both reported.

**Current results** (re-run the notebook to populate):

| Model | CV F1 | Test F1 | Test Recall |
|---|---|---|---|
| FLAML AutoML — default threshold | — | — | — |
| FLAML AutoML — tuned threshold | — | — | — |
| LSTM | — | — | — |

*Restart kernel → Run All to reproduce.*

**Explainability (SHAP):**
SHAP TreeExplainer identifies the most predictive features for the Non-Healthy class:
- `sensor_14_mean` / `sensor_14_std` — core speed tracks late-stage degradation
- `sensor_11_mean` — HPC static pressure rises with compressor fouling
- `sensor_04_delta` — rate-of-change in LPT outlet temperature detects abrupt deterioration

Note: `norm_cycle` is **excluded** from features — it encodes lifecycle position
derived from the failure point and constitutes target leakage.

![SHAP global importance](assets/shap_global_importance.png)

---

### 6. Deployment

#### MLflow experiment tracking

All training runs are logged to the local MLflow Model Registry under
`turbofan-health-classification`. Each run records:
- Section 0 constants (window sizes, RUL threshold, k-means clusters, etc.)
- FLAML search configuration (time budget, metric, estimator list)
- Best estimator name and hyperparameters
- CV F1, test F1, test recall
- Trained model artifact with input/output schema

```bash
python -m mlflow ui
# Open http://127.0.0.1:5000
```

#### FastAPI inference service

The trained pipeline is exposed as a REST API. Raw C-MAPSS sensor files are
accepted directly — operating-condition normalisation, rolling feature extraction,
and min-max scaling all run server-side.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe + loaded model metadata |
| POST | `/predict_file` | Upload raw C-MAPSS `.txt` → health label + probability |

```bash
curl -X POST "http://localhost:8001/predict_file?unit_id=1" \
  -F "file=@data/raw/test_FD001.txt"
# → {"labels": ["Healthy"], "predictions": [0], "probabilities": [[0.94, 0.06]], ...}
```

#### Docker containerisation

```bash
# Local (no Docker)
uvicorn utils.inference_api:app --reload --port 8000

# Docker (recommended)
docker compose up --build
# Swagger UI → http://localhost:8001/docs
```

The `mlruns/` and `models/` directories are **mounted as read-only volumes** —
not baked into the image. Retrain on the host, then `docker compose restart` to
deploy the new model version without rebuilding the image.

#### CI/CD

Pushing to `main` automatically rebuilds and publishes the Docker inference image
via GitHub Actions (`.github/workflows/docker.yml`). The image is hosted on GitHub
Container Registry and pulled by `docker-compose.yml` at runtime.

```
git push → GitHub Actions → docker build → ghcr.io/…/turbofan-health-api:latest
```

Training remains a **local step** — run the notebook, commit the updated `mlruns/`,
then push. No retraining happens inside Docker.

---

## Project Structure

```
predictive-maintenance-rul/
├── PredictiveMaintenance_Training.ipynb   # Main notebook (full CRISP-DM pipeline)
├── requirements.txt                       # Full training dependencies (pinned)
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

- The Healthy/Non-Healthy threshold is searched data-driven but remains a proxy for
  true maintenance economics; a cost-sensitive objective weighting missed faults
  higher than false alarms would better align with operational priorities
- Distribution shift monitoring (PSI on rolling feature windows) is not yet
  implemented — a natural next step before production deployment
- Transformer-based sequence models (e.g. Temporal Fusion Transformer) may capture
  longer-range degradation dependencies than the stacked LSTM
- The k-means operating-condition clustering uses k=6 to match FD002/FD004's known
  regimes; data-driven k selection (elbow / silhouette) would be more principled for
  unseen datasets
