"""Sklearn preprocessing and model pipeline builders."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrisense.config import ProjectConfig, load_config


def split_feature_groups(
    feature_names: list[str],
    config: ProjectConfig | None = None,
) -> tuple[list[str], list[str]]:
    """Partition *feature_names* into nominal and numeric columns."""
    cfg = config or load_config()
    nominal = [c for c in feature_names if c in cfg.features.nominal]
    numeric = [c for c in feature_names if c not in nominal]
    return nominal, numeric


def build_preprocessor(
    nominal_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a ColumnTransformer for model pipelines."""
    transformers: list[tuple] = []

    if nominal_cols:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                nominal_cols,
            )
        )

    if numeric_cols:
        if scale_numeric:
            transformers.append(("numeric", StandardScaler(), numeric_cols))
        else:
            transformers.append(("numeric", "passthrough", numeric_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator,
    feature_names: list[str],
    config: ProjectConfig | None = None,
    scale_numeric: bool = False,
) -> Pipeline:
    """Wrap *estimator* in a preprocessing + model Pipeline."""
    nominal, numeric = split_feature_groups(feature_names, config)
    preprocessor = build_preprocessor(nominal, numeric, scale_numeric=scale_numeric)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
