# Predictive Maintenance — Turbofan Engine Health Classification

Multi-class health classification on the NASA C-MAPSS turbofan engine dataset,
demonstrating an end-to-end ML pipeline from raw sensor data to a deployable
inference API.

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

- 4 sub-datasets (FD001–FD004): varying operating conditions and fault modes
- 21 sensor channels + 3 operational settings per cycle
- Full run-to-failure trajectories for each engine unit

Dataset is downloaded automatically via `kagglehub` on first run.

---

## Project Structure

```
predictive-maintenance-rul/
├── PredictiveMaintenance_Training.ipynb   # Main notebook (full pipeline)
├── requirements.txt
├── assets/                                # Exported figures
│   ├── confusion_matrices.png
│   ├── lstm_training_history.png
│   ├── rul_class_distribution.png
│   ├── rul_distribution.png
│   ├── sensor_correlation.png
│   ├── sensor_degradation_trends.png
│   ├── shap_beeswarm_critical.png
│   ├── shap_global_importance.png
│   └── xgb_feature_importance.png
├── data/
│   └── raw/                               # Local data cache (gitignored — auto-downloaded)
├── models/                                # Saved model artefacts (gitignored — regenerate via notebook)
└── utils/
    ├── download_dataset.py                # kagglehub download helper
    ├── data_loader.py                     # C-MAPSS loader + RUL computation
    ├── feature_engineering.py             # Rolling-window feature engineering
    ├── ml_classification.py               # XGBoost + LSTM wrappers, evaluation
    ├── inference_api.py                   # FastAPI inference endpoint
    └── plot_style.py                      # Global plot theme and colour palettes
```

---

## Pipeline Overview

```
Data Acquisition  →  EDA  →  Preprocessing  →  Modelling  →  Evaluation  →  Explainability
```

| Section | What happens |
|---|---|
| 1. Data Acquisition | Download C-MAPSS via kagglehub; load FD001–FD004 |
| 2. EDA | Class distribution, sensor trends, correlation heatmap |
| 3. Preprocessing | RUL labelling, 30-cycle rolling features, min-max normalisation |
| 4. Model Definition | XGBoost pipeline; LSTM sequence model |
| 5. Cross-Validation | 5-fold stratified CV for XGBoost; early stopping for LSTM |
| 6. Evaluation | Test-set metrics (accuracy, F1-macro, F1-weighted) |
| 7. Visualisations | Confusion matrices, feature importance, LSTM training curves |
| 8. Explainability | SHAP TreeExplainer — beeswarm + global bar chart |
| 9. Summary | Key findings and limitations |

---

## Models

### XGBoost (primary)
- Input: flat rolling-feature vector (73 features per cycle)
- 14 sensors × 5 statistics (mean/std/min/max/delta) + norm_cycle + 3 op settings
- 5-fold stratified CV; selected on F1-weighted

### LSTM (optional — requires TensorFlow)
- Input: sensor sequences `(30 cycles × 73 features)` — same rolling features as XGBoost
- 2-layer LSTM (128 → 64 units) with dropout + early stopping + class-weighted training
- Captures temporal degradation patterns end-to-end

---

## Key Results

| Model | CV F1-weighted | Test F1-weighted | Test F1-macro | Test Accuracy |
|---|---|---|---|---|
| XGBoost | 0.9542 ± 0.0016 | 0.9571 | 0.9442 | 0.9577 |
| LSTM | — | 0.8784 | 0.8620 | 0.8795 |

XGBoost is the primary model. LSTM requires TensorFlow and is optional.

---

## Explainability

SHAP values identify the most predictive features for the **Critical** class:

- `sensor_14_mean` / `sensor_14_std` — core speed tracks late-stage degradation
- `sensor_11_mean` — HPC static pressure rises with compressor fouling
- `norm_cycle` — temporal progress is independently predictive
- `sensor_04_delta` — rate-of-change in LPT outlet temperature

![SHAP global importance](assets/shap_global_importance.png)

---

## Setup

**Requirements:** Python 3.11, pip

```bash
pip install -r requirements.txt
```

**Kaggle API token** — required for automatic dataset download:

```bash
# Place kaggle.json in your home directory (created via Kaggle → Settings → API → New Token)
# Linux / macOS
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle
Copy-Item kaggle.json $env:USERPROFILE\.kaggle\
```

**Run the notebook:**

```bash
jupyter lab PredictiveMaintenance_Training.ipynb
```

---

## Inference API

A FastAPI endpoint is provided for real-time health prediction:

```bash
uvicorn utils.inference_api:app --reload
# GET  /health            →  liveness probe
# POST /predict/xgboost   →  { "health_class": 0, "health_label": "Healthy", "confidence": 0.97, "probabilities": [...] }
# POST /predict/lstm      →  same schema (sequence input required)
```

---

## Limitations & Next Steps

- Health-class thresholds (RUL 80/30) are heuristic; cost-sensitive optimisation
  could align them with actual maintenance economics
- FD002/FD004 (6 operating conditions) are harder sub-tasks; clustering operating
  regimes before classification may improve performance
- Transformer-based sequence models (e.g. TST) may outperform LSTM with less tuning
