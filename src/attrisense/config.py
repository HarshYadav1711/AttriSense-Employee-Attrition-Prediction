"""Configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from attrisense.utils.paths import CONFIG_PATH


@dataclass(frozen=True)
class ProjectConfig:
    """Typed view of ``configs/config.yaml``."""

    random_state: int
    raw_filename: str
    processed_filename: str
    target_column: str
    positive_class: str
    negative_class: str
    drop_columns: list[str]
    id_column: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        return cls(
            random_state=data["random_state"],
            raw_filename=data["data"]["raw_filename"],
            processed_filename=data["data"]["processed_filename"],
            target_column=data["target"]["column"],
            positive_class=data["target"]["positive_class"],
            negative_class=data["target"]["negative_class"],
            drop_columns=list(data.get("drop_columns", [])),
            id_column=data.get("id_column", "EmployeeNumber"),
        )


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load and parse the project YAML configuration."""
    config_path = path or CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return ProjectConfig.from_dict(raw)
