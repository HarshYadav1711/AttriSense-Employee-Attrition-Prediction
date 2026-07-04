"""Prediction helpers for the Streamlit application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from attrisense.inference import (
    PredictionResult,
    ShapExplanation,
    build_prediction_dataframe,
    compute_shap_explanation,
    get_input_columns,
    predict_attrition,
    validate_dataframe,
    validate_employee_record,
)


def default_employee_record(categorical_options: dict) -> dict:
    """Return sensible defaults for the single-employee form."""
    return {
        "Education": 3,
        "EnvironmentSatisfaction": 3,
        "JobInvolvement": 3,
        "JobSatisfaction": 3,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "WorkLifeBalance": 3,
        "BusinessTravel": categorical_options["BusinessTravel"][1],
        "Department": categorical_options["Department"][1],
        "EducationField": categorical_options["EducationField"][1],
        "Gender": categorical_options["Gender"][0],
        "JobRole": categorical_options["JobRole"][3],
        "MaritalStatus": categorical_options["MaritalStatus"][1],
        "OverTime": "No",
        "Age": 35,
        "DailyRate": 800,
        "DistanceFromHome": 5,
        "HourlyRate": 65,
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 12,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 2,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }


@st.cache_data(show_spinner=False)
def run_batch_prediction(upload_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Cached batch prediction for identical uploads."""
    result = predict_attrition(upload_df, threshold=threshold)
    output = build_prediction_dataframe(result)
    if "EmployeeNumber" in upload_df.columns:
        output.insert(0, "EmployeeNumber", upload_df["EmployeeNumber"].values)
    return output


def run_single_prediction(record: dict, threshold: float) -> tuple[PredictionResult, pd.DataFrame]:
    """Predict attrition for one employee record."""
    df = pd.DataFrame([record])
    df.insert(0, "EmployeeNumber", record.get("EmployeeNumber", 1))
    result = predict_attrition(df, threshold=threshold)
    return result, build_prediction_dataframe(result)


@st.cache_data(show_spinner="Computing SHAP explanation…")
def get_shap_explanation(feature_values: bytes, row_index: int = 0) -> ShapExplanation:
    """Cached SHAP computation keyed on feature values."""
    import io

    feature_frame = pd.read_parquet(io.BytesIO(feature_values))
    return compute_shap_explanation(feature_frame, row_index=row_index)


def feature_frame_cache_key(feature_frame: pd.DataFrame) -> bytes:
    """Serialize feature frame for SHAP cache key."""
    import io

    buffer = io.BytesIO()
    feature_frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def validate_single(record: dict):
    return validate_employee_record(record)


def validate_batch(df: pd.DataFrame):
    return validate_dataframe(df)


def input_schema_dataframe() -> pd.DataFrame:
    """Document required upload columns."""
    return pd.DataFrame({"RequiredColumn": get_input_columns()})
