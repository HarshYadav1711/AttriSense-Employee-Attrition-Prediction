"""Configuration loading."""

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
class ProjectConfig:
    """Typed view of ``configs/config.yaml``."""

    random_state: int
    raw_filename: str
    processed_filename: str
    preprocessed_filename: str
    target_column: str
    positive_class: str
    negative_class: str
    drop_columns: list[str]
    id_column: str
    features: FeatureConfig
    preprocessing: PreprocessingConfig

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
        return cls(
            random_state=data["random_state"],
            raw_filename=data["data"]["raw_filename"],
            processed_filename=data["data"]["processed_filename"],
            preprocessed_filename=data["data"]["preprocessed_filename"],
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
        )


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load and parse the project YAML configuration."""
    config_path = path or CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return ProjectConfig.from_dict(raw)
