"""Data access layer."""

from attrisense.data.cleaning import (
    clean_raw_data,
    drop_uninformative_columns,
    load_processed_data,
    save_processed_data,
    validate_no_missing_values,
    validate_target_values,
)
from attrisense.data.features import FEATURE_REGISTRY, feature_registry_dataframe
from attrisense.data.loader import dataset_summary, load_raw_data
from attrisense.data.preprocessing import (
    PreprocessingReport,
    build_feature_transformer,
    encode_target,
    handle_duplicates,
    inspect_duplicates,
    inspect_missing_values,
    inspect_outliers_iqr,
    load_feature_transformer,
    load_preprocessed_data,
    run_preprocessing_pipeline,
    save_preprocessing_artifacts,
    transform_features,
)

__all__ = [
    "FEATURE_REGISTRY",
    "PreprocessingReport",
    "build_feature_transformer",
    "clean_raw_data",
    "dataset_summary",
    "drop_uninformative_columns",
    "encode_target",
    "feature_registry_dataframe",
    "handle_duplicates",
    "inspect_duplicates",
    "inspect_missing_values",
    "inspect_outliers_iqr",
    "load_feature_transformer",
    "load_preprocessed_data",
    "load_processed_data",
    "load_raw_data",
    "run_preprocessing_pipeline",
    "save_preprocessing_artifacts",
    "save_processed_data",
    "transform_features",
    "validate_no_missing_values",
    "validate_target_values",
]
