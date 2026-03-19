"""utils/data_loader.py

pandas-based data loading for the NASA C-MAPSS turbofan engine dataset.

Each sub-dataset (FD001–FD004) is a space-delimited text file with 26 columns
(no header): unit number, cycle index, 3 operational settings, 21 sensor readings.

Public API::

    from utils.data_loader import load_all_fds, compute_rul
    df  = load_all_fds(data_dir)
    df  = compute_rul(df)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

COLUMN_NAMES: list[str] = (
    ["unit_nr", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

# Sensors that carry discriminative information (near-constant sensors excluded)
USEFUL_SENSORS: list[str] = [
    "sensor_02", "sensor_03", "sensor_04", "sensor_07",
    "sensor_08", "sensor_09", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]

OP_SETTINGS: list[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]

# RUL classification thresholds (piecewise-linear degradation model)
RUL_CAP: int = 125
RUL_HEALTHY_THR: int = 80
RUL_CRITICAL_THR: int = 30

CLASS_NAMES: list[str] = ["Healthy", "Degrading", "Critical"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_one(data_dir: str | Path, fd_id: int, split: str) -> pd.DataFrame:
    """Load a single C-MAPSS split file into a pandas DataFrame."""
    file_path = Path(data_dir) / f"{split}_FD{fd_id:03d}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES + ["_drop1", "_drop2"],
        engine="python",
    )
    df = df[COLUMN_NAMES].copy()
    df["fd_id"] = fd_id
    return df


def load_all_fds(
    data_dir: str | Path,
    fds: list[int] | None = None,
    split: str = "train",
) -> pd.DataFrame:
    """Load and concatenate multiple C-MAPSS sub-datasets.

    Args:
        data_dir: Directory containing the raw .txt files.
        fds:      Sub-dataset indices to load (default: [1, 2, 3, 4]).
        split:    'train' or 'test'.

    Returns:
        pandas DataFrame with all sub-datasets concatenated.
        'unit_nr' is globally unique via fd_id * 1000 + unit_nr.
    """
    if fds is None:
        fds = [1, 2, 3, 4]

    df = pd.concat([_load_one(data_dir, fd, split) for fd in fds], ignore_index=True)
    df["unit_nr"] = df["fd_id"] * 1000 + df["unit_nr"]
    return df


# ---------------------------------------------------------------------------
# RUL and health-class computation
# ---------------------------------------------------------------------------

def compute_rul(
    df: pd.DataFrame,
    rul_cap: int = RUL_CAP,
    healthy_thr: int = RUL_HEALTHY_THR,
    critical_thr: int = RUL_CRITICAL_THR,
) -> pd.DataFrame:
    """Add RUL (capped) and health_class columns to the training DataFrame.

    Args:
        df:           Raw training DataFrame (must have unit_nr, time_cycles).
        rul_cap:      Maximum RUL value (plateau assumption for early lifecycle).
        healthy_thr:  RUL threshold separating Healthy from Degrading.
        critical_thr: RUL threshold separating Degrading from Critical.

    Returns:
        DataFrame with added 'rul', 'capped_rul', 'health_class' columns.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["rul"]        = (max_cycles - df["time_cycles"]).astype(int)
    df["capped_rul"] = df["rul"].clip(upper=rul_cap).astype(int)
    df["health_class"] = np.where(
        df["capped_rul"] > healthy_thr, 0,
        np.where(df["capped_rul"] > critical_thr, 1, 2)
    ).astype(int)
    return df


def attach_test_rul(
    df_test: pd.DataFrame,
    data_dir: str | Path,
    fd_id: int,
    rul_cap: int = RUL_CAP,
    healthy_thr: int = RUL_HEALTHY_THR,
    critical_thr: int = RUL_CRITICAL_THR,
) -> pd.DataFrame:
    """Attach ground-truth RUL labels to a test DataFrame.

    Args:
        df_test:  Test DataFrame for a single FD split (unit_nr globally encoded).
        data_dir: Directory containing RUL_FD00X.txt files.
        fd_id:    Sub-dataset index matching df_test.
        rul_cap:  RUL cap for piecewise-linear model.
        healthy_thr:  Healthy/Degrading boundary.
        critical_thr: Degrading/Critical boundary.

    Returns:
        Test DataFrame with 'rul', 'capped_rul', 'health_class' columns.
    """
    rul_path = Path(data_dir) / f"RUL_FD{fd_id:03d}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_path}")

    true_rul = pd.read_csv(rul_path, header=None, names=["true_rul"]).squeeze()

    df_test   = df_test.copy()
    raw_unit  = (df_test["unit_nr"] % 1000).astype(int)
    max_cycles = df_test.groupby("unit_nr")["time_cycles"].transform("max")
    last_rul   = true_rul.values[raw_unit.values - 1]

    df_test["rul"]        = (last_rul + max_cycles.values - df_test["time_cycles"].values).astype(int)
    df_test["capped_rul"] = df_test["rul"].clip(upper=rul_cap).astype(int)
    df_test["health_class"] = np.where(
        df_test["capped_rul"] > healthy_thr, 0,
        np.where(df_test["capped_rul"] > critical_thr, 1, 2)
    ).astype(int)
    return df_test
