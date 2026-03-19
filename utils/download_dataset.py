"""utils/download_dataset.py

Download the NASA C-MAPSS turbofan engine degradation dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

The dataset contains four sub-datasets (FD001–FD004) with varying operating
conditions and fault modes.  Each is a multi-variate time-series of engine
sensor readings collected until failure.

Usage::

    from utils.download_dataset import download_cmapss
    data_dir = download_cmapss()
"""

import os
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected file structure for the NASA C-MAPSS dataset
# ---------------------------------------------------------------------------

_EXPECTED_FILES: list[str] = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]


def _find_data_root(base: Path) -> Path | None:
    """Recursively search for the directory that contains train_FD001.txt.

    Args:
        base: Root directory to begin the search.

    Returns:
        Path to the directory containing the dataset, or None if not found.
    """
    if (base / "train_FD001.txt").exists():
        return base
    for child in base.rglob("train_FD001.txt"):
        return child.parent
    return None


def download_cmapss(target_dir: str = "data/raw") -> Path:
    """Download the NASA C-MAPSS dataset via kagglehub and verify its contents.

    Requires a valid Kaggle API token (~/.kaggle/kaggle.json).
    The download is cached by kagglehub; re-running returns the cached path.

    Args:
        target_dir: Local directory where a symlink / copy reference is printed.
                    The actual files live inside the kagglehub cache.

    Returns:
        Path to the directory containing the 12 C-MAPSS .txt files.

    Raises:
        ImportError: If kagglehub is not installed.
        FileNotFoundError: If the expected files are absent after download.
    """
    try:
        import kagglehub  # lazy import — not needed at module load time
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the dataset.\n"
            "Install it with:  pip install kagglehub"
        ) from exc

    Path(target_dir).mkdir(parents=True, exist_ok=True)

    print("Downloading NASA C-MAPSS dataset from Kaggle (cached after first run)…")
    raw_path = Path(kagglehub.dataset_download("behrad3d/nasa-cmaps"))
    print(f"  kagglehub cache path : {raw_path}")

    data_root = _find_data_root(raw_path)
    if data_root is None:
        raise FileNotFoundError(
            f"C-MAPSS files not found under {raw_path}.\n"
            "Check your Kaggle credentials and internet connection."
        )

    # Verify that every expected file is present
    missing = [f for f in _EXPECTED_FILES if not (data_root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"The following expected files are missing from {data_root}:\n"
            + "\n".join(f"  {f}" for f in missing)
        )

    print(f"  Dataset root         : {data_root}")
    print(f"  Files verified       : {len(_EXPECTED_FILES)} / {len(_EXPECTED_FILES)}")
    return data_root


def check_local_data(data_dir: str = "data/raw") -> Path | None:
    """Check whether the C-MAPSS files already exist in a local directory.

    Useful for offline / pre-downloaded environments (e.g., Databricks DBFS).

    Args:
        data_dir: Directory to check.

    Returns:
        Path to the data directory if all files exist, otherwise None.
    """
    data_path = Path(data_dir)
    if all((data_path / f).exists() for f in _EXPECTED_FILES):
        print(f"Local data found at: {data_path}")
        return data_path
    return None


def get_data_dir(local_dir: str = "data/raw") -> Path:
    """Return the data directory, downloading from Kaggle only if necessary.

    First checks for a local copy; falls back to kagglehub download.

    Args:
        local_dir: Path to check for pre-downloaded data.

    Returns:
        Resolved Path to the C-MAPSS dataset directory.
    """
    local = check_local_data(local_dir)
    if local is not None:
        return local
    return download_cmapss(local_dir)
