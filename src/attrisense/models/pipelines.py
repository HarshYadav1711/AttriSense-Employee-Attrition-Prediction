"""Sklearn preprocessing and classifier pipeline builders.

Builds ``Pipeline`` objects that pair a ``ColumnTransformer`` (one-hot encoding
for nominal fields; optional ``StandardScaler`` for numeric fields) with a
classifier. Used by ``models.training`` during GridSearchCV and persisted
as a single joblib artifact for inference.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrisense.config import ProjectConfig, load_config


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return post-preprocessing feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def split_feature_groups(
    feature_names: list[str],
    config: ProjectConfig | None = None,
) -> tuple[list[str], list[str]]:
    """Split feature names into nominal and numeric groups per config.

    Nominal columns are one-hot encoded; all others pass through as numeric
    (optionally scaled for linear models).
    """
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
    """Return a fitted-ready ``Pipeline(preprocessor → classifier)``.

    Args:
        estimator: Sklearn-compatible classifier (unfitted).
        feature_names: Columns present in ``X`` before preprocessing.
        config: Project configuration for nominal column lookup.
        scale_numeric: When True, apply ``StandardScaler`` to numeric columns
            (required for Logistic Regression, unnecessary for tree models).
    """
    nominal, numeric = split_feature_groups(feature_names, config)
    preprocessor = build_preprocessor(nominal, numeric, scale_numeric=scale_numeric)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
