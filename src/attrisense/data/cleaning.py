"""Data cleaning pipeline for the employee attrition dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from attrisense.config import ProjectConfig, load_config
from attrisense.utils.paths import DATA_PROCESSED_DIR


def drop_uninformative_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Remove constant or otherwise uninformative columns.

    The IBM dataset includes fields like ``EmployeeCount`` (always 1),
    ``Over18`` (always Y), and ``StandardHours`` (always 80). These carry
    no signal and would add noise if one-hot encoded.
    """
    cfg = config or load_config()
    to_drop = columns if columns is not None else cfg.drop_columns
    existing = [col for col in to_drop if col in df.columns]
    return df.drop(columns=existing)


def validate_no_missing_values(df: pd.DataFrame) -> None:
    """Raise if any column contains null values."""
    missing = df.isna().sum()
    cols_with_nulls = missing[missing > 0]
    if not cols_with_nulls.empty:
        raise ValueError(f"Unexpected missing values found:\n{cols_with_nulls.to_dict()}")


def validate_target_values(df: pd.DataFrame, config: ProjectConfig | None = None) -> None:
    """Ensure the target column only contains expected class labels."""
    cfg = config or load_config()
    allowed = {cfg.positive_class, cfg.negative_class}
    observed = set(df[cfg.target_column].unique())
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected target values {unexpected}. Expected {allowed}."
        )


def clean_raw_data(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Apply the full cleaning pipeline and return a copy.

    Steps:
        1. Drop uninformative constant columns
        2. Validate no missing values
        3. Validate target column values
    """
    cfg = config or load_config()
    cleaned = df.copy()
    cleaned = drop_uninformative_columns(cleaned, config=cfg)
    validate_no_missing_values(cleaned)
    validate_target_values(cleaned, config=cfg)
    return cleaned


def save_processed_data(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Persist cleaned data as Parquet for downstream notebooks."""
    cfg = config or load_config()
    out_dir = output_dir or DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cfg.processed_filename
    df.to_parquet(path, index=False)
    return path


def load_processed_data(
    config: ProjectConfig | None = None,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Load previously cleaned data from disk."""
    cfg = config or load_config()
    in_dir = input_dir or DATA_PROCESSED_DIR
    path = in_dir / cfg.processed_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run the data cleaning notebook first."
        )
    return pd.read_parquet(path)
