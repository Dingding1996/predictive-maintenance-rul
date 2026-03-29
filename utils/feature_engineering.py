"""utils/feature_engineering.py

pandas-based feature engineering for NASA C-MAPSS.

Feature families:
  - Rolling statistics   : mean, std, min, max over a configurable window
  - Delta features       : first-difference (rate-of-change) per sensor
  - Long-window mean     : slow-trend rolling mean over a wider window (default 60)
  - Slope (momentum)     : short_mean − long_mean; positive = sensor rising recently
  - Cycle normalisation  : time_cycles normalised to [0, 1] per unit
  - Min-max scaling      : fit on training data, applied to any split

Usage::

    from utils.feature_engineering import (
        add_rolling_features_spark, add_long_window_features, get_feature_cols,
    )
    df_feat = add_rolling_features_spark(df, sensor_cols=USEFUL_SENSORS, window_size=30)
    df_feat = add_long_window_features(df_feat, sensor_cols=USEFUL_SENSORS)
    feature_cols = get_feature_cols(USEFUL_SENSORS, include_long_window=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_WINDOW_SIZE: int = 30
LONG_WINDOW_SIZE: int   = 60
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


def add_long_window_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    long_window: int = LONG_WINDOW_SIZE,
    short_window_col_suffix: str = "mean",
) -> pd.DataFrame:
    """Add slow-trend and momentum features using a longer rolling window.

    Must be called **after** ``add_rolling_features_spark`` so that the
    ``{col}_mean`` short-window columns already exist in ``df``.

    Three feature families added per sensor:

    * ``{col}_lw_mean``  — rolling mean over ``long_window`` cycles; captures
      gradual degradation trends that are invisible in a 30-cycle window.
    * ``{col}_lw_std``   — rolling std over ``long_window`` cycles; sensor variance
      tends to increase during degradation, and a wider window reveals this earlier.
    * ``{col}_slope``    — momentum proxy: ``short_mean − lw_mean``.
      Positive → sensor recently above its long-term level (rising trend).
      Negative → sensor recently below long-term level (declining / degrading).
      Avoids expensive OLS computation while conveying directional drift.

    Args:
        df:                     Input DataFrame (must already contain ``{col}_mean``
                                columns from ``add_rolling_features_spark``).
        sensor_cols:            Sensor column names (un-suffixed, e.g. ``sensor_02``).
        long_window:            Slow-trend window length in cycles (default 100).
        short_window_col_suffix: Suffix of the short-window mean column (default
                                 ``'mean'``; change only if a custom suffix was used).

    Returns:
        DataFrame with ``{col}_lw_mean`` and ``{col}_slope`` columns appended.
    """
    df = df.sort_values(["unit_nr", "time_cycles"]).copy()

    for col in sensor_cols:
        grouped = df.groupby("unit_nr")[col]

        # Slow-trend mean — uses a wider window to track baseline degradation
        lw_mean = grouped.transform(
            lambda s, w=long_window: s.rolling(w, min_periods=1).mean()
        ).astype(np.float32)
        df[f"{col}_lw_mean"] = lw_mean

        # Long-window std — variance tends to increase during degradation;
        # 100-cycle std reveals this trend earlier than the 30-cycle version
        df[f"{col}_lw_std"] = grouped.transform(
            lambda s, w=long_window: s.rolling(w, min_periods=1).std(ddof=1).fillna(0)
        ).astype(np.float32)

        # Slope proxy: short_mean − long_mean (direction of recent drift)
        short_mean_col = f"{col}_{short_window_col_suffix}"
        df[f"{col}_slope"] = (df[short_mean_col] - lw_mean).astype(np.float32)

    return df


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
# Operating-condition normalisation
# ---------------------------------------------------------------------------

def fit_condition_normaliser(
    df_train: pd.DataFrame,
    op_cols: list[str],
    sensor_cols: list[str],
    n_clusters: int = 6,
    random_state: int = 42,
) -> tuple:
    """Fit k-means operating-condition clusters and per-cluster z-score statistics.

    For datasets with multiple operating regimes (e.g. C-MAPSS FD002/FD004),
    sensor readings shift substantially across flight conditions.  This function
    identifies operating-condition clusters and computes per-cluster mean/std —
    fit on training data only to prevent leakage into the test split.

    Args:
        df_train:    Training DataFrame containing op_cols and sensor_cols.
        op_cols:     Operational-setting columns used for cluster assignment.
        sensor_cols: Sensor columns to z-score normalise.
        n_clusters:  Number of operating-condition clusters (k for k-means).
        random_state: Random seed for k-means reproducibility.

    Returns:
        Tuple (kmeans, cluster_stats):
          - kmeans       : Fitted KMeans object.
          - cluster_stats: Dict {cluster_id: {col: (mean, std)}}.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(df_train[op_cols].values)

    labels = kmeans.predict(df_train[op_cols].values)
    cluster_stats: dict = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_stats[c] = {
            col: (
                float(df_train.loc[mask, col].mean()),
                float(df_train.loc[mask, col].std(ddof=1)) + 1e-8,
            )
            for col in sensor_cols
        }
    return kmeans, cluster_stats


def apply_condition_normaliser(
    df: pd.DataFrame,
    kmeans,
    cluster_stats: dict,
    op_cols: list[str],
    sensor_cols: list[str],
) -> pd.DataFrame:
    """Apply pre-fit operating-condition z-score normalisation to a DataFrame.

    Assigns each row to the nearest operating-condition cluster (using the
    pre-fit k-means) and standardises each sensor column within its cluster
    using the training-split mean and std.

    Args:
        df:            DataFrame to normalise (train or test split).
        kmeans:        Fitted KMeans from fit_condition_normaliser.
        cluster_stats: Per-cluster stats dict from fit_condition_normaliser.
        op_cols:       Operational-setting columns for cluster assignment.
        sensor_cols:   Sensor columns to normalise.

    Returns:
        DataFrame with sensor columns replaced by within-cluster z-scores.
    """
    df = df.copy()
    labels = kmeans.predict(df[op_cols].values)
    df["_cond_cluster"] = labels

    for c, stats in cluster_stats.items():
        mask = df["_cond_cluster"] == c
        if not mask.any():
            continue
        for col in sensor_cols:
            mean, std = stats[col]
            df.loc[mask, col] = ((df.loc[mask, col] - mean) / std).astype(np.float32)

    return df.drop(columns=["_cond_cluster"])


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

def _get_new_feature_cols(
    sensor_cols: list[str],
    include_delta: bool,
    include_long_window: bool = False,
) -> list[str]:
    cols: list[str] = []
    for c in sensor_cols:
        for suf in ["mean", "std", "min", "max"]:
            cols.append(f"{c}_{suf}")
        if include_delta:
            cols.append(f"{c}_delta")
        if include_long_window:
            cols.append(f"{c}_lw_mean")
            cols.append(f"{c}_lw_std")
            cols.append(f"{c}_slope")
    return cols


def get_feature_cols(
    sensor_cols: list[str],
    include_delta: bool = True,
    include_norm_cycle: bool = True,
    include_op_settings: bool = True,
    op_settings: list[str] | None = None,
    include_long_window: bool = False,
) -> list[str]:
    """Return the complete ordered list of feature column names for modelling.

    Args:
        sensor_cols:          Sensor columns used during feature engineering.
        include_delta:        Whether delta features were computed.
        include_norm_cycle:   Whether norm_cycle was added.
        include_op_settings:  Whether operational setting columns are included.
        op_settings:          Names of operational-setting columns.
        include_long_window:  Whether long-window mean and slope features were
                              added via ``add_long_window_features``.

    Returns:
        Ordered list of feature column names.
    """
    feature_cols: list[str] = []
    if include_op_settings:
        feature_cols += (op_settings or ["op_setting_1", "op_setting_2", "op_setting_3"])
    if include_norm_cycle:
        feature_cols.append("norm_cycle")
    feature_cols += _get_new_feature_cols(sensor_cols, include_delta, include_long_window)
    return feature_cols
