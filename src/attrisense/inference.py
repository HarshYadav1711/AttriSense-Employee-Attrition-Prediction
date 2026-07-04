"""Inference helpers for attrition prediction and SHAP explanations.

Provides the runtime path from raw employee input (29 base columns) through
feature engineering to model scoring. Used by the Streamlit app and available
for programmatic batch scoring:

    from attrisense.inference import predict_attrition, build_prediction_dataframe
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from attrisense.config import ProjectConfig, load_config
from attrisense.data.feature_engineering import (
    FeatureEngineeringState,
    apply_engineered_features,
)
from attrisense.models.evaluation import get_transformed_feature_names
from attrisense.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR


@dataclass
class ValidationIssue:
    """Single validation error or warning."""

    field: str
    message: str
    severity: str  # "error" | "warning"


@dataclass
class PredictionResult:
    """Outcome of a single or batch prediction."""

    employee_ids: list[str | int]
    probabilities: np.ndarray
    predictions: np.ndarray
    risk_tiers: list[str]
    confidence_scores: np.ndarray
    feature_frame: pd.DataFrame


@dataclass
class ShapExplanation:
    """SHAP values for one prediction."""

    feature_names: list[str]
    shap_values: np.ndarray
    base_value: float
    feature_values: np.ndarray


INPUT_COLUMNS: list[str] = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

RISK_TIERS = (
    (0.60, "High"),
    (0.40, "Elevated"),
    (0.25, "Moderate"),
    (0.0, "Low"),
)


def load_best_model(models_dir: Path | None = None) -> Pipeline:
    """Load the persisted best-model pipeline."""
    path = (models_dir or MODELS_DIR) / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Best model not found at {path}. Run model training and evaluation first."
        )
    return joblib.load(path)


def load_feature_state(models_dir: Path | None = None) -> FeatureEngineeringState:
    """Load feature-engineering state for inference-time transforms."""
    path = (models_dir or MODELS_DIR) / "feature_engineering_state.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature engineering state not found at {path}. Run feature engineering first."
        )
    return joblib.load(path)


def load_selected_features(models_dir: Path | None = None) -> list[str]:
    """Load the model feature list."""
    path = (models_dir or MODELS_DIR) / "selected_features.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["selected_features"])


def risk_tier(probability: float) -> str:
    """Map attrition probability to an HR-facing risk label."""
    for threshold, label in RISK_TIERS:
        if probability >= threshold:
            return label
    return "Low"


def prediction_confidence(probability: float) -> float:
    """Distance from uncertain (0.5); higher means more decisive."""
    return float(abs(probability - 0.5) * 2)


def validate_employee_record(
    record: dict[str, Any],
    config: ProjectConfig | None = None,
) -> list[ValidationIssue]:
    """Validate a single employee input record."""
    cfg = config or load_config()
    issues: list[ValidationIssue] = []

    required = set(INPUT_COLUMNS)
    for field in required:
        value = record.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(ValidationIssue(field, "Required field is missing.", "error"))

    likert_1_4 = [
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    for field in likert_1_4:
        value = record.get(field)
        if value is not None and not (1 <= int(value) <= 4):
            issues.append(ValidationIssue(field, "Must be between 1 and 4.", "error"))

    if record.get("Education") is not None and not (1 <= int(record["Education"]) <= 5):
        issues.append(ValidationIssue("Education", "Must be between 1 and 5.", "error"))

    if record.get("StockOptionLevel") is not None and not (
        0 <= int(record["StockOptionLevel"]) <= 3
    ):
        issues.append(ValidationIssue("StockOptionLevel", "Must be between 0 and 3.", "error"))

    if record.get("PerformanceRating") is not None and int(record["PerformanceRating"]) not in (
        3,
        4,
    ):
        issues.append(
            ValidationIssue(
                "PerformanceRating",
                "Must be 3 or 4 (values observed in training data).",
                "error",
            )
        )

    if record.get("Age") is not None and not (18 <= int(record["Age"]) <= 70):
        issues.append(ValidationIssue("Age", "Must be between 18 and 70.", "error"))

    if record.get("MonthlyIncome") is not None and int(record["MonthlyIncome"]) <= 0:
        issues.append(ValidationIssue("MonthlyIncome", "Must be greater than zero.", "error"))

    tenure_checks = [
        ("YearsInCurrentRole", "YearsAtCompany", "Role tenure cannot exceed company tenure."),
        ("YearsWithCurrManager", "YearsAtCompany", "Manager tenure cannot exceed company tenure."),
        ("YearsSinceLastPromotion", "YearsAtCompany", "Years since promotion cannot exceed company tenure."),
        ("YearsAtCompany", "TotalWorkingYears", "Company tenure cannot exceed total working years."),
    ]
    for left, right, message in tenure_checks:
        if record.get(left) is not None and record.get(right) is not None:
            if int(record[left]) > int(record[right]):
                issues.append(ValidationIssue(left, message, "error"))

    if record.get("OverTime") not in (None, "Yes", "No"):
        issues.append(ValidationIssue("OverTime", "Must be 'Yes' or 'No'.", "error"))

    nominal = cfg.features.nominal
    for field in nominal:
        value = record.get(field)
        if value is not None and field in nominal and isinstance(value, str) and not value.strip():
            issues.append(ValidationIssue(field, "Invalid category.", "error"))

    return issues


def validate_dataframe(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate a batch upload DataFrame."""
    issues: list[ValidationIssue] = []
    missing_cols = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(
            ValidationIssue(
                "columns",
                f"Missing required columns: {', '.join(missing_cols)}",
                "error",
            )
        )
        return issues

    if df.empty:
        issues.append(ValidationIssue("rows", "Upload contains no data rows.", "error"))
        return issues

    for idx, row in df.iterrows():
        row_issues = validate_employee_record(row.to_dict())
        for issue in row_issues:
            issues.append(
                ValidationIssue(
                    f"row_{idx + 1}.{issue.field}",
                    issue.message,
                    issue.severity,
                )
            )
        if len([i for i in issues if i.severity == "error"]) >= 20:
            issues.append(ValidationIssue("rows", "Additional row errors omitted.", "warning"))
            break

    return issues


def prepare_feature_matrix(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply feature engineering and select model columns."""
    cfg = config or load_config()
    state = load_feature_state(models_dir)
    features = load_selected_features(models_dir)

    working = df[INPUT_COLUMNS].copy()
    featured = apply_engineered_features(working, state, cfg)
    return featured[features].copy()


def predict_attrition(
    df: pd.DataFrame,
    employee_id_col: str | None = "EmployeeNumber",
    config: ProjectConfig | None = None,
    models_dir: Path | None = None,
    threshold: float = 0.5,
) -> PredictionResult:
    """Run attrition prediction on one or more employee records."""
    pipeline = load_best_model(models_dir)
    feature_frame = prepare_feature_matrix(df, config, models_dir)

    probabilities = pipeline.predict_proba(feature_frame)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    ids: list[str | int]
    if employee_id_col and employee_id_col in df.columns:
        ids = df[employee_id_col].tolist()
    else:
        ids = list(range(1, len(df) + 1))

    return PredictionResult(
        employee_ids=ids,
        probabilities=probabilities,
        predictions=predictions,
        risk_tiers=[risk_tier(p) for p in probabilities],
        confidence_scores=np.array([prediction_confidence(p) for p in probabilities]),
        feature_frame=feature_frame,
    )


def build_prediction_dataframe(result: PredictionResult) -> pd.DataFrame:
    """Format prediction output for display and CSV export."""
    return pd.DataFrame(
        {
            "EmployeeID": result.employee_ids,
            "AttritionProbability": np.round(result.probabilities, 4),
            "PredictedAttrition": np.where(result.predictions == 1, "Yes", "No"),
            "RiskTier": result.risk_tiers,
            "ConfidenceScore": np.round(result.confidence_scores, 4),
        }
    )


def compute_shap_explanation(
    feature_frame: pd.DataFrame,
    row_index: int = 0,
    models_dir: Path | None = None,
    background_size: int = 120,
) -> ShapExplanation:
    """Compute SHAP values for a single prediction using the linear model."""
    pipeline = load_best_model(models_dir)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    cfg = load_config()
    featured_path = DATA_PROCESSED_DIR / cfg.feature_engineered_filename
    if featured_path.exists():
        background_df = pd.read_parquet(featured_path)
        background_features = load_selected_features(models_dir)
        background_sample = background_df[background_features].sample(
            n=min(background_size, len(background_df)),
            random_state=cfg.random_state,
        )
    else:
        background_sample = feature_frame.iloc[[row_index]]

    background_transformed = preprocessor.transform(background_sample)
    row_transformed = preprocessor.transform(feature_frame.iloc[[row_index]])
    feature_names = get_transformed_feature_names(pipeline)

    explainer = shap.LinearExplainer(
        classifier,
        background_transformed,
        feature_names=feature_names,
    )
    shap_values = explainer.shap_values(row_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return ShapExplanation(
        feature_names=feature_names,
        shap_values=np.asarray(shap_values).flatten(),
        base_value=float(np.asarray(explainer.expected_value).flatten()[0]),
        feature_values=np.asarray(row_transformed).flatten(),
    )


def shap_contributions_table(explanation: ShapExplanation, top_n: int = 12) -> pd.DataFrame:
    """Return top SHAP contributors as a sorted DataFrame."""
    df = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "shap_value": explanation.shap_values,
            "feature_value": explanation.feature_values,
        }
    )
    df["abs_shap"] = df["shap_value"].abs()
    df["direction"] = np.where(df["shap_value"] >= 0, "Increases risk", "Decreases risk")
    return df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
