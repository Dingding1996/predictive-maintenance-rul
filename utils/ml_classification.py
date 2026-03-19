"""utils/ml_classification.py

Machine learning models for turbofan engine health classification.

Two model families are provided:
  1. FLAML AutoML — automated model selection + hyperparameter search on tabular
                    rolling features (sklearn-compatible interface, MLflow tracking)
  2. LSTM          — sequence classifier on raw sensor windows (Keras / TensorFlow)

Both are trained to predict one of three health classes:
  0 = Healthy | 1 = Degrading | 2 = Critical

Usage::

    from utils.ml_classification import (
        run_automl_with_mlflow,
        build_lstm_model, train_lstm,
        evaluate_classification, get_confusion_matrix,
        save_model, load_model,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

# ---------------------------------------------------------------------------
# FLAML custom metric — f1_weighted
# ---------------------------------------------------------------------------
# FLAML 2.3.x built-ins do not include 'f1_weighted'; we supply it as a
# callable.  Signature required by FLAML: returns (val_loss, metrics_dict)
# where val_loss is minimised (so loss = 1 − f1_weighted).

def _flaml_f1_weighted(
    X_val, y_val, estimator, labels,
    X_train, y_train,
    weight_val=None, weight_train=None,
    *args,
):
    """Custom FLAML metric: minimise 1 − F1-weighted."""
    y_pred    = estimator.predict(X_val)
    f1w       = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    val_loss  = 1.0 - f1w
    return val_loss, {"f1_weighted": f1w}

# Lazy import TensorFlow so the module loads even without TF installed
_tf_available: bool = False
try:
    import tensorflow as tf
    from tensorflow.keras import callbacks as keras_callbacks
    from tensorflow.keras import layers, models
    _tf_available = True
except ImportError:
    warnings.warn(
        "TensorFlow not found — LSTM model is unavailable.  "
        "Install it with: pip install tensorflow",
        ImportWarning,
        stacklevel=2,
    )

CLASS_NAMES: list[str] = ["Healthy", "Degrading", "Critical"]
N_CLASSES: int = len(CLASS_NAMES)


# ===========================================================================
# AutoML (FLAML + MLflow)
# ===========================================================================

def run_automl_with_mlflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    time_budget: int = 300,
    metric: str = "f1_weighted",
    experiment_name: str = "automl-experiment",
    tracking_uri: str = "./mlruns",
    extra_params: dict | None = None,
    random_state: int = 42,
    scale_params: dict | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Run FLAML AutoML with MLflow experiment tracking.

    Searches over multiple model families (LightGBM, XGBoost, Random Forest,
    Extra Trees, linear models) and logs the best result to MLflow as a
    registered model artifact.

    Args:
        X_train:         Training feature matrix (min-max normalised).
        y_train:         Integer-encoded class labels.
        time_budget:     Maximum search time in seconds (FLAML budget).
        metric:          Optimisation metric passed to FLAML
                         ('f1_weighted', 'accuracy', 'log_loss', etc.).
        experiment_name: MLflow experiment name (created if absent).
        tracking_uri:    MLflow tracking URI. Use a local path (e.g. './mlruns')
                         for file-based tracking — no server required.
        extra_params:    Additional key-value pairs to log as MLflow params
                         (pass Section 0 constants here for full reproducibility).
        random_state:    Random seed forwarded to FLAML.
        scale_params:    Min-max scale parameters from training preprocessing
                         (col → (min, max)).  If provided, saved as a JSON artifact
                         in the MLflow run and to ``models/scale_params.json`` on disk
                         for use by the inference API raw-data endpoint.

    Returns:
        Tuple of:
          - Fitted FLAML AutoML object.
              · ``automl.predict(X)``         — inference on new data.
              · ``automl.best_estimator``     — winning model family name.
              · ``automl.best_config``        — best hyperparameter config.
              · ``automl.model.estimator``    — raw sklearn estimator (for SHAP).
          - cv_results DataFrame with per-estimator best F1-weighted scores,
            sorted descending. Mirrors the ``cv_results`` convention used
            elsewhere in the notebook.

    Raises:
        ImportError: If ``flaml`` or ``mlflow`` are not installed.
    """
    try:
        from flaml import AutoML
    except ImportError:
        raise ImportError(
            "FLAML is required for AutoML: pip install flaml[default]"
        )
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        raise ImportError("MLflow is required: pip install mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="flaml_automl") as run:

        # --- Log all Section 0 constants + search configuration ---
        search_params: dict[str, Any] = {
            "time_budget_s": time_budget,
            "metric":        metric,
            "random_state":  random_state,
        }
        if extra_params:
            search_params.update(extra_params)
        mlflow.log_params(search_params)

        # --- Run FLAML automated search ---
        # Use the custom callable for f1_weighted (not a FLAML 2.3.x built-in).
        # Fall back to the string for any other metric that FLAML supports natively.
        flaml_metric = _flaml_f1_weighted if metric == "f1_weighted" else metric

        automl = AutoML()
        automl.fit(
            X_train, y_train,
            task="classification",
            metric=flaml_metric,
            time_budget=time_budget,
            seed=random_state,
            verbose=1,
        )

        # --- Log best results ---
        best_f1 = 1.0 - automl.best_loss   # FLAML minimises loss = 1 − metric
        mlflow.log_metric("cv_f1_weighted_best", best_f1)
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_param("best_config",    str(automl.best_config))

        # --- Register the underlying sklearn estimator in MLflow ---
        # Infer input/output schema from a small training sample so the MLflow
        # UI shows the feature schema and output type under "Schema".
        from mlflow.models import infer_signature
        sample     = X_train[:500] if len(X_train) > 500 else X_train
        y_sample   = automl.predict(sample)
        signature  = infer_signature(sample, y_sample)

        # automl.model.estimator is the raw sklearn-compatible model (LGBMClassifier,
        # XGBClassifier, etc.) — compatible with mlflow.sklearn and SHAP TreeExplainer.
        mlflow.sklearn.log_model(
            sk_model=automl.model.estimator,
            artifact_path="best_model",
            registered_model_name=experiment_name,
            signature=signature,
        )

        # --- Persist scale_params for the inference raw-data endpoint ---
        # Saved to both MLflow artifacts (for audit) and models/scale_params.json
        # (for Docker mount access without a running MLflow server).
        if scale_params is not None:
            import json, tempfile
            from pathlib import Path as _Path
            sp_serialisable = {k: list(v) for k, v in scale_params.items()}
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(sp_serialisable, tmp)
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="scale_params")
            _Path(tmp_path).unlink(missing_ok=True)
            # Also write to models/ for direct file-based loading in Docker
            models_dir = _Path(tracking_uri.lstrip("./")) if tracking_uri.startswith("./") else _Path("models")
            models_dir = _Path("models")
            models_dir.mkdir(exist_ok=True)
            (models_dir / "scale_params.json").write_text(json.dumps(sp_serialisable))
            print(f"  scale_params    : saved to models/scale_params.json ({len(scale_params)} cols)")

        print(f"\n{'=' * 50}")
        print(f"  FLAML AutoML — Search Complete")
        print(f"{'=' * 50}")
        print(f"  MLflow run ID   : {run.info.run_id}")
        print(f"  Best estimator  : {automl.best_estimator}")
        print(f"  CV F1-weighted  : {best_f1:.4f}  (1 − best_loss)")
        print(f"  Best config     : {automl.best_config}")

    # --- Build cv_results summary (per-estimator best scores) ---
    cv_results = _build_cv_results(automl)

    return automl, cv_results


def _build_cv_results(automl: Any) -> pd.DataFrame:
    """Extract per-estimator best F1-weighted scores from a fitted FLAML object.

    Falls back to a single-row summary if FLAML's internal attribute is absent
    (API varies across versions).

    Args:
        automl: Fitted FLAML AutoML object.

    Returns:
        DataFrame with columns ['estimator', 'f1_weighted'], sorted descending.
    """
    if hasattr(automl, "best_config_per_estimator"):
        rows = []
        for est, info in automl.best_config_per_estimator.items():
            if info is None:
                continue
            # FLAML 2.3.x: info is a dict with key "val_loss"
            # Earlier versions: info is a tuple (loss, config, n_iter, model)
            if isinstance(info, dict):
                loss = info.get("val_loss")
            else:
                loss = info[0] if info else None
            if loss is not None:
                rows.append({
                    "estimator":   est,
                    "f1_weighted": round(1.0 - loss, 4),
                })
        if rows:
            return (
                pd.DataFrame(rows)
                .sort_values("f1_weighted", ascending=False)
                .reset_index(drop=True)
            )

    # Fallback — single-row summary from top-level attributes
    return pd.DataFrame({
        "estimator":   [automl.best_estimator],
        "f1_weighted": [round(1.0 - automl.best_loss, 4)],
    })


# ===========================================================================
# LSTM
# ===========================================================================

def build_lstm_model(
    seq_len: int,
    n_features: int,
    n_classes: int = N_CLASSES,
    lstm_units: list[int] | None = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> "tf.keras.Model":  # type: ignore[name-defined]
    """Build a stacked LSTM model for sequence health classification.

    Architecture:
      LSTM(units[0]) → Dropout → LSTM(units[1], return_sequences=False) →
      Dropout → Dense(64, relu) → Dense(n_classes, softmax)

    Args:
        seq_len:       Number of timesteps per input sequence.
        n_features:    Number of sensor/feature channels per timestep.
        n_classes:     Number of output health classes.
        lstm_units:    LSTM hidden sizes for each stacked layer (default [128, 64]).
        dropout_rate:  Dropout fraction applied after each LSTM layer.
        learning_rate: Adam optimiser learning rate.
        random_state:  NumPy/TF seed for weight initialisation.

    Returns:
        Compiled Keras Model.

    Raises:
        RuntimeError: If TensorFlow is not installed.
    """
    if not _tf_available:
        raise RuntimeError("TensorFlow is required for LSTM.  pip install tensorflow")

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    if lstm_units is None:
        lstm_units = [128, 64]

    inp = layers.Input(shape=(seq_len, n_features), name="sensor_sequence")

    x = layers.LSTM(lstm_units[0], return_sequences=True, name="lstm_1")(inp)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    for i, units in enumerate(lstm_units[1:], start=2):
        return_seq = i < len(lstm_units)  # only last LSTM returns False
        x = layers.LSTM(units, return_sequences=return_seq, name=f"lstm_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    x = layers.Dense(64, activation="relu", name="fc_1")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="LSTM_HealthClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm(
    model: "tf.keras.Model",  # type: ignore[name-defined]
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,
    class_weight: dict[int, float] | None = None,
) -> dict[str, list[float]]:
    """Train an LSTM model with early stopping and learning-rate reduction.

    Args:
        model:        Compiled Keras model (from build_lstm_model).
        X_train:      Training sequences, shape (n, seq_len, n_features).
        y_train:      Training labels, shape (n,), integer class indices.
        X_val:        Validation sequences.
        y_val:        Validation labels.
        epochs:       Maximum training epochs.
        batch_size:   Mini-batch size.
        patience:     Early stopping patience (epochs without val_loss improvement).
        class_weight: Optional dict mapping class index → weight. Pass the output
                      of sklearn compute_class_weight to counter label imbalance.

    Returns:
        Keras history dict {'loss', 'accuracy', 'val_loss', 'val_accuracy'}.
    """
    if not _tf_available:
        raise RuntimeError("TensorFlow is required for LSTM.  pip install tensorflow")

    cb_list = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=patience // 2,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        class_weight=class_weight,
        verbose=1,
    )
    return history.history


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    model_name: str = "Model",
) -> dict[str, float]:
    """Compute and print standard multi-class classification metrics.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        class_names: Human-readable class names (index → name).
        model_name:  Label used in printed output.

    Returns:
        Dict with keys 'accuracy', 'f1_macro', 'f1_weighted'.
    """
    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Test-Set Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """Return a confusion matrix as a labelled pandas DataFrame.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        class_names: Row / column labels.

    Returns:
        DataFrame with true labels as rows and predicted labels as columns.
    """
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


# ===========================================================================
# Model persistence
# ===========================================================================

def save_model(
    model: Any,
    path: str | Path,
    model_type: str = "lstm",
) -> None:
    """Save a trained model to disk.

    Note: FLAML AutoML / sklearn models are registered via MLflow in Section 5
    and do not require this function. Use this for the Keras LSTM only.

    For sklearn-compatible models: uses joblib (pickle-compatible .pkl).
    For Keras LSTM models:         uses model.save() (.keras format).

    Args:
        model:      Fitted model or Pipeline to save.
        path:       Target file path (extension ignored for Keras — .keras added).
        model_type: 'sklearn' or 'lstm'.

    Raises:
        ValueError: If model_type is not recognised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "sklearn":
        save_path = path.with_suffix(".pkl")
        joblib.dump(model, save_path)
        print(f"sklearn model saved → {save_path}")

    elif model_type == "lstm":
        if not _tf_available:
            raise RuntimeError("TensorFlow required to save LSTM model.")
        save_path = path.with_suffix(".keras")
        model.save(str(save_path))
        print(f"LSTM model saved → {save_path}")

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'sklearn' or 'lstm'.")


def load_model(path: str | Path, model_type: str = "lstm") -> Any:
    """Load a previously saved model from disk.

    Args:
        path:       Path to the saved model file.
        model_type: 'sklearn' or 'lstm'.

    Returns:
        Loaded model (sklearn-compatible for 'sklearn', Keras Model for 'lstm').

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If model_type is not recognised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if model_type == "sklearn":
        model = joblib.load(path)
        print(f"sklearn model loaded ← {path}")
        return model

    elif model_type == "lstm":
        if not _tf_available:
            raise RuntimeError("TensorFlow required to load LSTM model.")
        model = models.load_model(str(path))
        print(f"LSTM model loaded ← {path}")
        return model

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'sklearn' or 'lstm'.")
