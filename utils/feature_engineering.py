"""utils/feature_engineering.py

pandas-based feature engineering for NASA C-MAPSS.

Feature families:
  - Rolling statistics   : mean, std, min, max over a configurable window
  - Delta features       : first-difference (rate-of-change) per sensor
  - Cycle normalisation  : time_cycles normalised to [0, 1] per unit
  - Min-max scaling      : fit on training data, applied to any split

Usage::

    from utils.dsp_features import add_rolling_features, get_feature_cols
    df_feat = add_rolling_features(df, sensor_cols=USEFUL_SENSORS, window_size=30)
    feature_cols = get_feature_cols(USEFUL_SENSORS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW_SIZE: int = 30
STAT_SUFFIXES: list[str] = ["mean", "std", "min", "max", "delta"]


# ---------------------------------------------------------------------------
# Rolling feature extraction
# ---------------------------------------------------------------------------

def add_rolling_features_spark(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    include_delta: bool = True,
    **_kwargs,
) -> pd.DataFrame:
    """Add rolling statistical features for each sensor.

    Computes mean, std, min, max (and optionally delta) over a rolling window
    of `window_size` cycles per engine unit.

    Args:
        df:            Input DataFrame (must have unit_nr, time_cycles).
        sensor_cols:   Sensor column names to process.
        window_size:   Rolling window length (inclusive of current row).
        include_delta: Whether to compute first-difference delta features.

    Returns:
        DataFrame with additional rolling-feature columns appended.
    """
    df = df.sort_values(["unit_nr", "time_cycles"]).copy()

    for col in sensor_cols:
        grouped = df.groupby("unit_nr")[col]
        # Use w=window_size default arg to capture value, not reference
        df[f"{col}_mean"] = grouped.transform(
            lambda s, w=window_size: s.rolling(w, min_periods=1).mean()
        ).astype(np.float32)
        df[f"{col}_std"] = grouped.transform(
            lambda s, w=window_size: s.rolling(w, min_periods=1).std(ddof=1).fillna(0)
        ).astype(np.float32)
        df[f"{col}_min"] = grouped.transform(
            lambda s, w=window_size: s.rolling(w, min_periods=1).min()
        ).astype(np.float32)
        df[f"{col}_max"] = grouped.transform(
            lambda s, w=window_size: s.rolling(w, min_periods=1).max()
        ).astype(np.float32)
        if include_delta:
            df[f"{col}_delta"] = (
                grouped.transform(lambda s: s.diff()).fillna(0).astype(np.float32)
            )

    return df


# Alias so existing notebook calls work unchanged
add_rolling_features = add_rolling_features_spark


# ---------------------------------------------------------------------------
# Cycle normalisation
# ---------------------------------------------------------------------------

def add_cycle_normalisation_spark(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
    """Add a normalised cycle counter [0, 1] per unit.

    Args:
        df: DataFrame with unit_nr and time_cycles columns.

    Returns:
        DataFrame with an added 'norm_cycle' column (float32, [0, 1]).
    """
    df = df.copy()
    max_cycles    = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["norm_cycle"] = (df["time_cycles"] / max_cycles).astype(np.float32)
    return df


add_cycle_normalisation = add_cycle_normalisation_spark


# ---------------------------------------------------------------------------
# Min-max normalisation
# ---------------------------------------------------------------------------

def min_max_normalise_spark(
    df: pd.DataFrame,
    feature_cols: list[str],
    **_kwargs,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Min-max normalise feature columns to [0, 1].

    Call only on the TRAINING split to avoid data leakage.

    Args:
        df:           Training DataFrame.
        feature_cols: Feature column names to normalise.

    Returns:
        Tuple of:
          - Normalised DataFrame.
          - scale_params dict: col → (min_val, max_val).
    """
    df = df.copy()
    scale_params: dict[str, tuple[float, float]] = {
        col: (float(df[col].min()), float(df[col].max()))
        for col in feature_cols
    }
    df = apply_min_max_spark(df, scale_params, feature_cols)
    return df, scale_params


min_max_normalise = min_max_normalise_spark


def apply_min_max_spark(
    df: pd.DataFrame,
    scale_params: dict[str, tuple[float, float]],
    feature_cols: list[str],
    **_kwargs,
) -> pd.DataFrame:
    """Apply pre-computed min-max scale parameters to a DataFrame.

    Args:
        df:           DataFrame to normalise.
        scale_params: Dict from min_max_normalise_spark (col → (min, max)).
        feature_cols: Feature columns to normalise.

    Returns:
        DataFrame with normalised feature columns clipped to [0, 1].
    """
    df = df.copy()
    for col in feature_cols:
        col_min, col_max = scale_params[col]
        denom    = col_max - col_min if col_max != col_min else 1.0
        df[col]  = ((df[col] - col_min) / denom).clip(0.0, 1.0).astype(np.float32)
    return df


apply_min_max = apply_min_max_spark


# ---------------------------------------------------------------------------
# LSTM sequence construction
# ---------------------------------------------------------------------------

def create_lstm_sequences(
    df: pd.DataFrame,
    sensor_cols: list[str],
    label_col: str = "health_class",
    seq_len: int = DEFAULT_WINDOW_SIZE,
    unit_col: str = "unit_nr",
    cycle_col: str = "time_cycles",
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sliding-window sequences for LSTM training.

    For each engine unit, sequences of length `seq_len` are extracted.
    The label is the health class at the last timestep of each window.

    Args:
        df:          DataFrame with unit/cycle/sensor columns.
        sensor_cols: Feature columns to include in the sequence.
        label_col:   Integer health class column (0 / 1 / 2).
        seq_len:     Timesteps per sequence.
        unit_col:    Engine unit identifier column.
        cycle_col:   Cycle ordering column.

    Returns:
        Tuple (X, y):
          - X : np.ndarray  (n_sequences, seq_len, n_features), float32
          - y : np.ndarray  (n_sequences,), int32
    """
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _, unit_df in df.groupby(unit_col):
        unit_df     = unit_df.sort_values(cycle_col).reset_index(drop=True)
        sensor_vals = unit_df[sensor_cols].values.astype(np.float32)
        label_vals  = unit_df[label_col].values.astype(np.int32)

        for i in range(seq_len, len(sensor_vals) + 1):
            sequences.append(sensor_vals[i - seq_len : i])
            labels.append(label_vals[i - 1])

    return np.stack(sequences, axis=0), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_new_feature_cols(sensor_cols: list[str], include_delta: bool) -> list[str]:
    cols: list[str] = []
    for c in sensor_cols:
        for suf in ["mean", "std", "min", "max"]:
            cols.append(f"{c}_{suf}")
        if include_delta:
            cols.append(f"{c}_delta")
    return cols


def get_feature_cols(
    sensor_cols: list[str],
    include_delta: bool = True,
    include_norm_cycle: bool = True,
    include_op_settings: bool = True,
    op_settings: list[str] | None = None,
) -> list[str]:
    """Return the complete ordered list of feature column names for modelling.

    Args:
        sensor_cols:         Sensor columns used during feature engineering.
        include_delta:       Whether delta features were computed.
        include_norm_cycle:  Whether norm_cycle was added.
        include_op_settings: Whether operational setting columns are included.
        op_settings:         Names of operational-setting columns.

    Returns:
        Ordered list of feature column names.
    """
    feature_cols: list[str] = []
    if include_op_settings:
        feature_cols += (op_settings or ["op_setting_1", "op_setting_2", "op_setting_3"])
    if include_norm_cycle:
        feature_cols.append("norm_cycle")
    feature_cols += _get_new_feature_cols(sensor_cols, include_delta)
    return feature_cols
