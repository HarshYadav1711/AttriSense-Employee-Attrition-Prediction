"""Project configuration loading from YAML.

``configs/config.yaml`` is the single source of truth for paths, feature
typing, preprocessing rules, feature-engineering parameters, and modeling
defaults. All pipeline stages accept an optional ``ProjectConfig``; when
omitted, ``load_config()`` reads from the repository root automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from attrisense.utils.paths import CONFIG_PATH


@dataclass(frozen=True)
class FeatureConfig:
    """Feature grouping for encoding and scaling."""

    ordinal: list[str]
    nominal: list[str]
    continuous: list[str]
    scale_when_required: list[str]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Outlier inspection settings."""

    outlier_iqr_multiplier: float
    cap_outliers: bool


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Feature engineering parameters."""

    satisfaction_columns: list[str]
    burnout_wlb_threshold: int
    correlation_threshold: float


@dataclass(frozen=True)
class ModelingConfig:
    """Model training parameters."""

    test_size: float
    cv_folds: int
    scoring: str
    n_jobs: int


@dataclass(frozen=True)
class ProjectConfig:
    """Typed view of ``configs/config.yaml``."""

    random_state: int
    raw_filename: str
    processed_filename: str
    preprocessed_filename: str
    feature_engineered_filename: str
    target_column: str
    positive_class: str
    negative_class: str
    drop_columns: list[str]
    id_column: str
    features: FeatureConfig
    preprocessing: PreprocessingConfig
    feature_engineering: FeatureEngineeringConfig
    modeling: ModelingConfig

    @property
    def model_feature_columns(self) -> list[str]:
        """All columns intended as model inputs (excludes target and id)."""
        return (
            self.features.ordinal
            + self.features.nominal
            + self.features.continuous
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        feat = data["features"]
        prep = data.get("preprocessing", {})
        fe = data.get("feature_engineering", {})
        mod = data.get("modeling", {})
        return cls(
            random_state=data["random_state"],
            raw_filename=data["data"]["raw_filename"],
            processed_filename=data["data"]["processed_filename"],
            preprocessed_filename=data["data"]["preprocessed_filename"],
            feature_engineered_filename=data["data"]["feature_engineered_filename"],
            target_column=data["target"]["column"],
            positive_class=data["target"]["positive_class"],
            negative_class=data["target"]["negative_class"],
            drop_columns=list(data.get("drop_columns", [])),
            id_column=data.get("id_column", "EmployeeNumber"),
            features=FeatureConfig(
                ordinal=list(feat["ordinal"]),
                nominal=list(feat["nominal"]),
                continuous=list(feat["continuous"]),
                scale_when_required=list(feat["scale_when_required"]),
            ),
            preprocessing=PreprocessingConfig(
                outlier_iqr_multiplier=float(prep.get("outlier_iqr_multiplier", 1.5)),
                cap_outliers=bool(prep.get("cap_outliers", False)),
            ),
            feature_engineering=FeatureEngineeringConfig(
                satisfaction_columns=list(
                    fe.get(
                        "satisfaction_columns",
                        [
                            "JobSatisfaction",
                            "EnvironmentSatisfaction",
                            "RelationshipSatisfaction",
                            "WorkLifeBalance",
                        ],
                    )
                ),
                burnout_wlb_threshold=int(fe.get("burnout_wlb_threshold", 2)),
                correlation_threshold=float(fe.get("correlation_threshold", 0.92)),
            ),
            modeling=ModelingConfig(
                test_size=float(mod.get("test_size", 0.2)),
                cv_folds=int(mod.get("cv_folds", 5)),
                scoring=str(mod.get("scoring", "roc_auc")),
                n_jobs=int(mod.get("n_jobs", -1)),
            ),
        )


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load and parse the project YAML configuration."""
    config_path = path or CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return ProjectConfig.from_dict(raw)
