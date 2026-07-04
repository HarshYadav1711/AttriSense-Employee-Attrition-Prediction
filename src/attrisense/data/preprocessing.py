"""Full preprocessing pipeline: inspect, clean, encode, and persist."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrisense.config import ProjectConfig, load_config
from attrisense.data.cleaning import (
    clean_raw_data,
    drop_uninformative_columns,
    validate_no_missing_values,
    validate_target_values,
)
from attrisense.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR


@dataclass
class DuplicateReport:
    """Result of duplicate inspection."""

    duplicate_rows: int
    duplicate_ids: int
    rows_removed: int
    action_taken: str


@dataclass
class MissingReport:
    """Result of missing-value inspection."""

    total_missing: int
    columns_with_missing: dict[str, int]
    action_taken: str


@dataclass
class OutlierReport:
    """IQR-based outlier flags per numeric column (diagnostic only)."""

    details: pd.DataFrame
    action_taken: str


@dataclass
class PreprocessingReport:
    """Aggregated log of every preprocessing decision."""

    input_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    preprocessed_shape: tuple[int, int]
    dropped_columns: list[str]
    duplicate_report: DuplicateReport
    missing_report: MissingReport
    outlier_report: OutlierReport
    encoding_summary: dict[str, Any]
    scaling_applied: bool
    output_paths: dict[str, str] = field(default_factory=dict)


def inspect_duplicates(
    df: pd.DataFrame,
    id_column: str,
) -> DuplicateReport:
    """Count duplicate rows and identifier collisions."""
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df[id_column].duplicated().sum()) if id_column in df.columns else 0

    if dup_rows == 0 and dup_ids == 0:
        action = "No duplicates found; no rows removed."
    else:
        action = (
            f"Would remove {dup_rows} duplicate row(s) and "
            f"{dup_ids} duplicate id(s) — review before dropping."
        )

    return DuplicateReport(
        duplicate_rows=dup_rows,
        duplicate_ids=dup_ids,
        rows_removed=0,
        action_taken=action,
    )


def handle_duplicates(
    df: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, DuplicateReport]:
    """Remove exact duplicate rows; keep first occurrence of each."""
    report = inspect_duplicates(df, id_column)
    cleaned = df.drop_duplicates(keep="first").copy()

    if id_column in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=[id_column], keep="first")

    rows_removed = len(df) - len(cleaned)
    report.rows_removed = rows_removed
    if rows_removed > 0:
        report.action_taken = f"Removed {rows_removed} duplicate row(s)."
    return cleaned, report


def inspect_missing_values(df: pd.DataFrame) -> MissingReport:
    """Summarise missing values per column."""
    missing = df.isna().sum()
    cols_with_missing = {
        col: int(count) for col, count in missing.items() if count > 0
    }
    total = int(missing.sum())

    if total == 0:
        action = "No missing values; imputation not applied."
    else:
        action = (
            f"{total} missing value(s) detected across "
            f"{len(cols_with_missing)} column(s); imputation required."
        )

    return MissingReport(
        total_missing=total,
        columns_with_missing=cols_with_missing,
        action_taken=action,
    )


def inspect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    multiplier: float = 1.5,
) -> OutlierReport:
    """Flag IQR outliers for numeric columns. Diagnostic only by default."""
    rows: list[dict[str, Any]] = []

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (series < lower) | (series > upper)
        n_outliers = int(mask.sum())
        unique_vals = int(series.nunique())

        rows.append(
            {
                "column": col,
                "min": series.min(),
                "max": series.max(),
                "q1": q1,
                "q3": q3,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "outlier_count": n_outliers,
                "outlier_pct": round(n_outliers / len(df) * 100, 2),
                "unique_values": unique_vals,
            }
        )

    details = pd.DataFrame(rows)
    action = (
        "Outlier flags recorded for review. No capping or row removal applied "
        "— flagged values fall within plausible HR ranges or are discrete counts."
    )
    return OutlierReport(details=details, action_taken=action)


def encode_target(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
    target_col_name: str = "Attrition_label",
) -> pd.DataFrame:
    """Map target to binary integer: 1 = Yes (left), 0 = No (stayed)."""
    cfg = config or load_config()
    result = df.copy()
    mapping = {cfg.positive_class: 1, cfg.negative_class: 0}
    result[target_col_name] = result[cfg.target_column].map(mapping)
    if result[target_col_name].isna().any():
        raise ValueError("Target encoding produced null values; check class labels.")
    return result


def build_feature_transformer(
    config: ProjectConfig | None = None,
    apply_scaling: bool = False,
) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for nominal one-hot + numeric passthrough."""
    cfg = config or load_config()
    nominal = cfg.features.nominal
    ordinal = cfg.features.ordinal
    continuous = cfg.features.continuous

    passthrough_cols = ordinal + continuous

    if apply_scaling:
        scale_cols = [
            c for c in cfg.features.scale_when_required if c in passthrough_cols
        ]
        no_scale_cols = [c for c in passthrough_cols if c not in scale_cols]
        transformers = [
            (
                "nominal",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                nominal,
            ),
            ("scale", StandardScaler(), scale_cols),
            ("pass", "passthrough", no_scale_cols),
        ]
    else:
        transformers = [
            (
                "nominal",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                nominal,
            ),
            ("numeric", "passthrough", passthrough_cols),
        ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_encoded_feature_names(
    transformer: ColumnTransformer,
    config: ProjectConfig | None = None,
) -> list[str]:
    """Return output column names after fitting the transformer."""
    cfg = config or load_config()
    try:
        return list(transformer.get_feature_names_out())
    except Exception:
        nominal_names = []
        encoder = transformer.named_transformers_["nominal"]
        if hasattr(encoder, "get_feature_names_out"):
            nominal_names = list(encoder.get_feature_names_out(cfg.features.nominal))
        numeric_names = cfg.features.ordinal + cfg.features.continuous
        return nominal_names + numeric_names


def transform_features(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
    apply_scaling: bool = False,
) -> tuple[pd.DataFrame, ColumnTransformer, dict[str, Any]]:
    """Encode categoricals and optionally scale continuous features."""
    cfg = config or load_config()
    feature_cols = cfg.model_feature_columns

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns not found: {missing_cols}")

    transformer = build_feature_transformer(cfg, apply_scaling=apply_scaling)
    matrix = transformer.fit_transform(df[feature_cols])
    feature_names = get_encoded_feature_names(transformer, cfg)

    encoded = pd.DataFrame(matrix, columns=feature_names, index=df.index)

    summary = {
        "nominal_columns": cfg.features.nominal,
        "ordinal_columns": cfg.features.ordinal,
        "continuous_columns": cfg.features.continuous,
        "one_hot_encoding": "drop='first' to reduce multicollinearity",
        "ordinal_treatment": "kept as integer (ordered scales)",
        "scaling_applied": apply_scaling,
        "output_feature_count": len(feature_names),
        "output_columns": feature_names,
    }

    return encoded, transformer, summary


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
    apply_scaling: bool = False,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingReport]:
    """Execute the full preprocessing pipeline.

    Steps:
        1. Handle duplicates
        2. Inspect missing values
        3. Drop constant / irrelevant columns
        4. Validate data quality
        5. Inspect outliers (diagnostic)
        6. Encode target and features
        7. Persist cleaned and preprocessed datasets

    Returns:
        Tuple of (cleaned_df, preprocessed_df, report).
    """
    cfg = config or load_config()
    input_shape = df.shape

    # Step 1 — duplicates
    deduped, dup_report = handle_duplicates(df, cfg.id_column)

    # Step 2 — missing values
    missing_report = inspect_missing_values(deduped)

    # Step 3 — drop irrelevant columns
    dropped = [c for c in cfg.drop_columns if c in deduped.columns]
    cleaned = drop_uninformative_columns(deduped, config=cfg)

    # Step 4 — validate
    validate_no_missing_values(cleaned)
    validate_target_values(cleaned, config=cfg)

    # Step 5 — outlier inspection
    outlier_cols = cfg.features.continuous + cfg.features.ordinal
    outlier_report = inspect_outliers_iqr(
        cleaned,
        outlier_cols,
        multiplier=cfg.preprocessing.outlier_iqr_multiplier,
    )

    # Step 6 — encoding
    cleaned_with_label = encode_target(cleaned, config=cfg)
    encoded_features, transformer, encoding_summary = transform_features(
        cleaned,
        config=cfg,
        apply_scaling=apply_scaling,
    )

    preprocessed = pd.concat(
        [
            cleaned[[cfg.id_column]].reset_index(drop=True),
            cleaned[[cfg.target_column]].reset_index(drop=True),
            cleaned_with_label[["Attrition_label"]].reset_index(drop=True),
            encoded_features.reset_index(drop=True),
        ],
        axis=1,
    )

    report = PreprocessingReport(
        input_shape=input_shape,
        cleaned_shape=cleaned.shape,
        preprocessed_shape=preprocessed.shape,
        dropped_columns=dropped,
        duplicate_report=dup_report,
        missing_report=missing_report,
        outlier_report=outlier_report,
        encoding_summary=encoding_summary,
        scaling_applied=apply_scaling,
    )

    if save_artifacts:
        paths = save_preprocessing_artifacts(
            cleaned,
            preprocessed,
            transformer,
            config=cfg,
        )
        report.output_paths = paths

    return cleaned, preprocessed, report


def save_preprocessing_artifacts(
    cleaned: pd.DataFrame,
    preprocessed: pd.DataFrame,
    transformer: ColumnTransformer,
    config: ProjectConfig | None = None,
    output_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, str]:
    """Write cleaned data, preprocessed matrix, and fitted transformer."""
    cfg = config or load_config()
    out_dir = output_dir or DATA_PROCESSED_DIR
    mdir = models_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / cfg.processed_filename
    preprocessed_path = out_dir / cfg.preprocessed_filename
    transformer_path = mdir / "feature_transformer.joblib"

    cleaned.to_parquet(cleaned_path, index=False)
    preprocessed.to_parquet(preprocessed_path, index=False)
    joblib.dump(transformer, transformer_path)

    return {
        "cleaned": str(cleaned_path),
        "preprocessed": str(preprocessed_path),
        "transformer": str(transformer_path),
    }


def load_preprocessed_data(
    config: ProjectConfig | None = None,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the model-ready preprocessed dataset."""
    cfg = config or load_config()
    in_dir = input_dir or DATA_PROCESSED_DIR
    path = in_dir / cfg.preprocessed_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {path}. Run the preprocessing pipeline first."
        )
    return pd.read_parquet(path)


def load_feature_transformer(models_dir: Path | None = None) -> ColumnTransformer:
    """Load the fitted feature transformer saved during preprocessing."""
    mdir = models_dir or MODELS_DIR
    path = mdir / "feature_transformer.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature transformer not found at {path}. Run the preprocessing pipeline first."
        )
    return joblib.load(path)
