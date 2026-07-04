"""Per-column metadata for inspection and documentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureKind = Literal["target", "identifier", "constant", "ordinal", "nominal", "continuous"]


@dataclass(frozen=True)
class FeatureSpec:
    """Descriptive metadata for a single dataset column."""

    name: str
    kind: FeatureKind
    description: str
    action: str
    justification: str


# Documented inspection result for every raw column (35 total).
FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="Age",
        kind="continuous",
        description="Employee age in years",
        action="Keep",
        justification="Valid range 18–60 in dataset; used as continuous numeric feature.",
    ),
    FeatureSpec(
        name="Attrition",
        kind="target",
        description="Whether the employee left the company",
        action="Keep; encode to 0/1",
        justification="Binary target variable for classification.",
    ),
    FeatureSpec(
        name="BusinessTravel",
        kind="nominal",
        description="Travel frequency: Non-Travel, Travel_Rarely, Travel_Frequently",
        action="Keep; one-hot encode",
        justification="Nominal category with 3 levels; no natural ordering.",
    ),
    FeatureSpec(
        name="DailyRate",
        kind="continuous",
        description="Daily rate of pay",
        action="Keep",
        justification="Numeric compensation field; scale only for linear models.",
    ),
    FeatureSpec(
        name="Department",
        kind="nominal",
        description="Department assignment",
        action="Keep; one-hot encode",
        justification="Nominal with 3 levels (HR, R&D, Sales).",
    ),
    FeatureSpec(
        name="DistanceFromHome",
        kind="continuous",
        description="Miles from home to workplace",
        action="Keep",
        justification="Integer distance 1–29; valid commute proxy.",
    ),
    FeatureSpec(
        name="Education",
        kind="ordinal",
        description="Education level coded 1–5",
        action="Keep as integer",
        justification="Ordered scale (Below College through Doctor); not one-hot encoded.",
    ),
    FeatureSpec(
        name="EducationField",
        kind="nominal",
        description="Field of study",
        action="Keep; one-hot encode",
        justification="Nominal with 6 categories; no inherent order.",
    ),
    FeatureSpec(
        name="EmployeeCount",
        kind="constant",
        description="Count of employees (always 1 per row)",
        action="Remove",
        justification="Single unique value (1); zero variance, no predictive signal.",
    ),
    FeatureSpec(
        name="EmployeeNumber",
        kind="identifier",
        description="Unique employee identifier",
        action="Keep in cleaned data; exclude from feature matrix",
        justification="Primary key for traceability; not a causal predictor.",
    ),
    FeatureSpec(
        name="EnvironmentSatisfaction",
        kind="ordinal",
        description="Environment satisfaction Likert 1–4",
        action="Keep as integer",
        justification="Ordered satisfaction scale.",
    ),
    FeatureSpec(
        name="Gender",
        kind="nominal",
        description="Gender: Male or Female",
        action="Keep; one-hot encode",
        justification="Nominal binary category; production use requires fairness review.",
    ),
    FeatureSpec(
        name="HourlyRate",
        kind="continuous",
        description="Hourly rate of pay",
        action="Keep",
        justification="Numeric compensation field.",
    ),
    FeatureSpec(
        name="JobInvolvement",
        kind="ordinal",
        description="Job involvement rating 1–4",
        action="Keep as integer",
        justification="Ordered engagement scale.",
    ),
    FeatureSpec(
        name="JobLevel",
        kind="ordinal",
        description="Job level 1–5",
        action="Keep as integer",
        justification="Ordered hierarchy level.",
    ),
    FeatureSpec(
        name="JobRole",
        kind="nominal",
        description="Job title role (9 categories)",
        action="Keep; one-hot encode",
        justification="Highest-cardinality nominal field; no natural ordering.",
    ),
    FeatureSpec(
        name="JobSatisfaction",
        kind="ordinal",
        description="Job satisfaction Likert 1–4",
        action="Keep as integer",
        justification="Ordered satisfaction scale.",
    ),
    FeatureSpec(
        name="MaritalStatus",
        kind="nominal",
        description="Marital status: Divorced, Married, Single",
        action="Keep; one-hot encode",
        justification="Nominal with 3 unordered categories.",
    ),
    FeatureSpec(
        name="MonthlyIncome",
        kind="continuous",
        description="Monthly income in currency units",
        action="Keep; no outlier capping",
        justification="IQR flags high earners but values are plausible; capping would remove signal.",
    ),
    FeatureSpec(
        name="MonthlyRate",
        kind="continuous",
        description="Monthly rate of pay",
        action="Keep",
        justification="Numeric compensation field.",
    ),
    FeatureSpec(
        name="NumCompaniesWorked",
        kind="continuous",
        description="Number of employers before current company",
        action="Keep; no outlier capping",
        justification="Range 0–9; values at upper bound reflect job mobility, not data errors.",
    ),
    FeatureSpec(
        name="Over18",
        kind="constant",
        description="Age over 18 indicator (always Y)",
        action="Remove",
        justification="Single unique value ('Y'); zero variance.",
    ),
    FeatureSpec(
        name="OverTime",
        kind="nominal",
        description="Whether employee works overtime: Yes/No",
        action="Keep; one-hot encode",
        justification="Binary nominal category strongly associated with attrition in HR literature.",
    ),
    FeatureSpec(
        name="PercentSalaryHike",
        kind="continuous",
        description="Last percentage salary increase",
        action="Keep",
        justification="Integer percent 11–25; valid range.",
    ),
    FeatureSpec(
        name="PerformanceRating",
        kind="ordinal",
        description="Performance rating (3 = Excellent, 4 = Outstanding)",
        action="Keep as integer; no outlier treatment",
        justification="Only two values (3, 4) in dataset; IQR 'outliers' are a statistical artifact.",
    ),
    FeatureSpec(
        name="RelationshipSatisfaction",
        kind="ordinal",
        description="Relationship satisfaction Likert 1–4",
        action="Keep as integer",
        justification="Ordered satisfaction scale.",
    ),
    FeatureSpec(
        name="StandardHours",
        kind="constant",
        description="Standard working hours (always 80)",
        action="Remove",
        justification="Single unique value (80); zero variance.",
    ),
    FeatureSpec(
        name="StockOptionLevel",
        kind="ordinal",
        description="Stock option level 0–3",
        action="Keep as integer",
        justification="Ordered benefit tier; level 3 is valid, not an error.",
    ),
    FeatureSpec(
        name="TotalWorkingYears",
        kind="continuous",
        description="Total years in workforce",
        action="Keep; no outlier capping",
        justification="Long-tenure employees (up to 40 years) are valid observations.",
    ),
    FeatureSpec(
        name="TrainingTimesLastYear",
        kind="continuous",
        description="Training sessions attended last year",
        action="Keep; no outlier capping",
        justification="Count 0–6; higher counts are valid, not erroneous.",
    ),
    FeatureSpec(
        name="WorkLifeBalance",
        kind="ordinal",
        description="Work-life balance rating 1–4",
        action="Keep as integer",
        justification="Ordered satisfaction scale.",
    ),
    FeatureSpec(
        name="YearsAtCompany",
        kind="continuous",
        description="Years with current employer",
        action="Keep; no outlier capping",
        justification="Tenure up to 40 years is plausible for senior staff.",
    ),
    FeatureSpec(
        name="YearsInCurrentRole",
        kind="continuous",
        description="Years in current job role",
        action="Keep",
        justification="Valid tenure range 0–18.",
    ),
    FeatureSpec(
        name="YearsSinceLastPromotion",
        kind="continuous",
        description="Years since last promotion",
        action="Keep",
        justification="Valid range 0–15; long gaps are informative for attrition.",
    ),
    FeatureSpec(
        name="YearsWithCurrManager",
        kind="continuous",
        description="Years with current manager",
        action="Keep",
        justification="Valid range 0–17.",
    ),
)


def feature_registry_dataframe():
    """Return the feature registry as a Pandas DataFrame for notebooks."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "column": spec.name,
                "kind": spec.kind,
                "description": spec.description,
                "action": spec.action,
                "justification": spec.justification,
            }
            for spec in FEATURE_REGISTRY
        ]
    )
