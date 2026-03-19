"""utils/ml_classification.py

Machine learning models for turbofan engine health classification.

Two model families are provided:
  1. XGBoost  — gradient-boosted trees on tabular rolling features (sklearn Pipeline)
  2. LSTM     — sequence classifier on raw sensor windows (Keras / TensorFlow)

Both are trained to predict one of three health classes:
  0 = Healthy | 1 = Degrading | 2 = Critical

Usage::

    from utils.ml_classification import (
        build_xgb_pipeline, train_xgb_cv,
        build_lstm_model, train_lstm,
        evaluate_classification, save_model, load_model,
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
# XGBoost
# ===========================================================================

def build_xgb_pipeline(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    random_state: int = 42,
    scale_pos_weight: float | None = None,
) -> Pipeline:
    """Build a sklearn Pipeline wrapping StandardScaler + XGBClassifier.

    Using a Pipeline ensures the scaler is re-fit inside each CV fold,
    preventing data leakage from test-set statistics.

    Args:
        n_estimators:      Number of boosting rounds.
        max_depth:         Maximum tree depth.
        learning_rate:     Shrinkage (eta) per round.
        subsample:         Row sub-sampling ratio per tree.
        colsample_bytree:  Column sub-sampling ratio per tree.
        min_child_weight:  Minimum sum of instance weight in a child node.
        random_state:      Global seed for reproducibility.
        scale_pos_weight:  Class weight for imbalanced datasets (or None).

    Returns:
        Unfitted sklearn Pipeline with steps ['scaler', 'model'].
    """
    xgb_params: dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        objective="multi:softmax",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    if scale_pos_weight is not None:
        xgb_params["scale_pos_weight"] = scale_pos_weight

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(**xgb_params)),
    ])
    return pipeline


def train_xgb_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame]:
    """Run stratified cross-validation on an XGBoost pipeline.

    Args:
        pipeline:     Unfitted sklearn Pipeline (from build_xgb_pipeline).
        X_train:      Training feature matrix.
        y_train:      Training labels (integer-encoded).
        n_splits:     Number of CV folds.
        random_state: Seed for StratifiedKFold shuffling.

    Returns:
        Tuple of:
          - `pipeline` re-fit on the full training set after CV.
          - `cv_results` DataFrame with per-fold accuracy and F1-weighted.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}

    raw = cross_validate(
        pipeline, X_train, y_train,
        cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False,
    )

    cv_results = pd.DataFrame({
        "fold":        list(range(1, n_splits + 1)),
        "accuracy":    raw["test_accuracy"],
        "f1_weighted": raw["test_f1_weighted"],
    })

    # Re-fit on full training data after CV evaluation
    pipeline.fit(X_train, y_train)

    print("\nXGBoost Cross-Validation Results:")
    print(cv_results.to_string(index=False))
    print(f"\n  Mean accuracy    : {cv_results['accuracy'].mean():.4f} "
          f"± {cv_results['accuracy'].std():.4f}")
    print(f"  Mean F1-weighted : {cv_results['f1_weighted'].mean():.4f} "
          f"± {cv_results['f1_weighted'].std():.4f}")

    return pipeline, cv_results


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
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
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
    model_type: str = "xgboost",
) -> None:
    """Save a trained model to disk.

    For XGBoost pipelines: uses joblib (pickle-compatible .pkl).
    For Keras LSTM models:  uses model.save() (.keras format).

    Args:
        model:      Fitted model or Pipeline to save.
        path:       Target file path (extension ignored for Keras — .keras added).
        model_type: 'xgboost' or 'lstm'.

    Raises:
        ValueError: If model_type is not recognised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "xgboost":
        save_path = path.with_suffix(".pkl")
        joblib.dump(model, save_path)
        print(f"XGBoost pipeline saved → {save_path}")

    elif model_type == "lstm":
        if not _tf_available:
            raise RuntimeError("TensorFlow required to save LSTM model.")
        save_path = path.with_suffix(".keras")
        model.save(str(save_path))
        print(f"LSTM model saved → {save_path}")

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'xgboost' or 'lstm'.")


def load_model(path: str | Path, model_type: str = "xgboost") -> Any:
    """Load a previously saved model from disk.

    Args:
        path:       Path to the saved model file.
        model_type: 'xgboost' or 'lstm'.

    Returns:
        Loaded model (sklearn Pipeline for XGBoost, Keras Model for LSTM).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If model_type is not recognised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if model_type == "xgboost":
        model = joblib.load(path)
        print(f"XGBoost pipeline loaded ← {path}")
        return model

    elif model_type == "lstm":
        if not _tf_available:
            raise RuntimeError("TensorFlow required to load LSTM model.")
        model = models.load_model(str(path))
        print(f"LSTM model loaded ← {path}")
        return model

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'xgboost' or 'lstm'.")
