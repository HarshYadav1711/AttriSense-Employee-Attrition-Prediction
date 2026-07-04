"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from attrisense.config import ProjectConfig, load_config
from attrisense.utils.paths import DATA_RAW_DIR


def load_raw_data(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Read the raw employee attrition CSV.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        DataFrame with one row per employee record.
    """
    cfg = config or load_config()
    path = DATA_RAW_DIR / cfg.raw_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download WA_Fn-UseC_-HR-Employee-Attrition.csv and place it in data/raw/."
        )
    return pd.read_csv(path)


def dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return a compact summary dict suitable for logging or notebook display."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
