"""Smoke tests for public-release readiness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


def test_package_imports() -> None:
    import attrisense  # noqa: F401
    from attrisense.config import load_config
    from attrisense.models.metrics import classification_metrics
    from attrisense.models.pipelines import get_transformed_feature_names

    cfg = load_config()
    assert cfg.random_state == 42
    assert classification_metrics is not None
    assert get_transformed_feature_names is not None


def test_config_paths_exist(project_root: Path) -> None:
    assert (project_root / "configs" / "config.yaml").is_file()
    assert (project_root / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv").is_file()


def test_input_columns_match_config(project_root: Path) -> None:
    from attrisense.config import load_config
    from attrisense.inference import get_input_columns

    columns = get_input_columns()
    cfg = load_config()
    selected_path = project_root / "models" / "selected_features.json"
    if selected_path.exists():
        dropped = set(json.loads(selected_path.read_text())["dropped_features"])
        expected = [c for c in cfg.model_feature_columns if c not in dropped]
        assert columns == expected


def test_training_results_use_relative_paths(project_root: Path) -> None:
    results_path = project_root / "models" / "training_results.json"
    if not results_path.exists():
        pytest.skip("training_results.json not generated yet")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for model in payload["models"]:
        path = model["model_path"]
        assert not Path(path).is_absolute()
        assert ":" not in path  # no Windows drive letters


def test_validate_employee_record() -> None:
    from attrisense.inference import get_input_columns, validate_employee_record

    record = {col: 1 for col in get_input_columns()}
    record.update(
        {
            "BusinessTravel": "Travel_Rarely",
            "Department": "Sales",
            "EducationField": "Life Sciences",
            "Gender": "Female",
            "JobRole": "Sales Executive",
            "MaritalStatus": "Single",
            "OverTime": "No",
            "Age": 35,
            "MonthlyIncome": 5000,
            "PerformanceRating": 3,
            "EnvironmentSatisfaction": 3,
            "JobInvolvement": 3,
            "JobSatisfaction": 3,
            "RelationshipSatisfaction": 3,
            "WorkLifeBalance": 3,
        }
    )
    issues = validate_employee_record(record)
    assert not any(i.severity == "error" for i in issues)
