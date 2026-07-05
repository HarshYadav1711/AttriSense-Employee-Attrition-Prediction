"""Feature engineering — business-motivated derived features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from attrisense.config import ProjectConfig, load_config
from attrisense.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR


@dataclass(frozen=True)
class EngineeredFeatureSpec:
    """Metadata for a single derived feature."""

    name: str
    formula: str
    business_explanation: str


ENGINEERED_FEATURE_SPECS: tuple[EngineeredFeatureSpec, ...] = (
    EngineeredFeatureSpec(
        name="promotion_stagnation_ratio",
        formula="YearsSinceLastPromotion / max(YearsAtCompany, 1)",
        business_explanation=(
            "Share of company tenure spent without a promotion. "
            "High values indicate career stagnation — a retention risk flagged in EDA "
            "via tenure and promotion-related correlations."
        ),
    ),
    EngineeredFeatureSpec(
        name="role_tenure_share",
        formula="YearsInCurrentRole / max(YearsAtCompany, 1)",
        business_explanation=(
            "Fraction of company tenure spent in the current role. "
            "Low values may indicate frequent internal moves; values near 1 suggest "
            "the employee has not changed role since joining."
        ),
    ),
    EngineeredFeatureSpec(
        name="manager_tenure_share",
        formula="YearsWithCurrManager / max(YearsAtCompany, 1)",
        business_explanation=(
            "Proportion of company tenure spent with the current manager. "
            "Manager relationship length is linked to attrition in correlation analysis."
        ),
    ),
    EngineeredFeatureSpec(
        name="income_vs_role_median",
        formula="MonthlyIncome / median(MonthlyIncome | JobRole)",
        business_explanation=(
            "Pay relative to the median for the same job role. "
            "Values below 1.0 suggest below-peer compensation — EDA showed "
            "lower absolute income aligns with higher attrition."
        ),
    ),
    EngineeredFeatureSpec(
        name="compensation_per_experience",
        formula="MonthlyIncome / max(TotalWorkingYears, 1)",
        business_explanation=(
            "Monthly pay per year of total work experience. "
            "Captures whether compensation keeps pace with career length."
        ),
    ),
    EngineeredFeatureSpec(
        name="avg_satisfaction_score",
        formula="mean(JobSatisfaction, EnvironmentSatisfaction, "
        "RelationshipSatisfaction, WorkLifeBalance)",
        business_explanation=(
            "Composite engagement score across four satisfaction dimensions. "
            "EDA showed each dimension predicts attrition; the average summarises "
            "overall sentiment without replacing individual scores."
        ),
    ),
    EngineeredFeatureSpec(
        name="burnout_risk_flag",
        formula="OverTime == 'Yes' AND WorkLifeBalance <= 2",
        business_explanation=(
            "Binary flag combining overtime and poor work-life balance — "
            "the two strongest operational attrition drivers from EDA "
            "(~31% attrition for each factor)."
        ),
    ),
    EngineeredFeatureSpec(
        name="job_stability_index",
        formula="TotalWorkingYears / max(NumCompaniesWorked, 1)",
        business_explanation=(
            "Average years per employer across the employee's career. "
            "Higher values indicate stable career patterns; lower values "
            "indicate frequent job changes before the current company."
        ),
    ),
)


@dataclass
class FeatureEngineeringState:
    """Statistics fit on training data for reproducible transforms."""

    role_income_medians: dict[str, float]
    engineered_columns: list[str] = field(
        default_factory=lambda: [s.name for s in ENGINEERED_FEATURE_SPECS]
    )


@dataclass
class RedundancyDecision:
    """Record of why a feature was kept or dropped."""

    dropped_feature: str
    kept_feature: str
    correlation: float
    dropped_importance: float
    kept_importance: float
    reason: str


@dataclass
class FeatureEngineeringReport:
    """Summary of feature engineering run."""

    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    engineered_columns: list[str]
    selected_features: list[str]
    dropped_features: list[str]
    redundancy_decisions: list[RedundancyDecision]
    importance_ranking: pd.DataFrame
    output_paths: dict[str, str] = field(default_factory=dict)


def engineered_feature_catalog() -> pd.DataFrame:
    """Return engineered feature definitions as a DataFrame."""
    return pd.DataFrame(
        [
            {
                "feature": spec.name,
                "formula": spec.formula,
                "business_explanation": spec.business_explanation,
            }
            for spec in ENGINEERED_FEATURE_SPECS
        ]
    )


def fit_feature_engineering_state(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> FeatureEngineeringState:
    """Compute role-level income medians from the reference dataset."""
    medians = df.groupby("JobRole")["MonthlyIncome"].median().to_dict()
    return FeatureEngineeringState(role_income_medians=medians)


def apply_engineered_features(
    df: pd.DataFrame,
    state: FeatureEngineeringState,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Add business-motivated derived columns to a copy of *df*."""
    cfg = config or load_config()
    result = df.copy()
    sat_cols = cfg.feature_engineering.satisfaction_columns
    wlb_threshold = cfg.feature_engineering.burnout_wlb_threshold

    years_at_co = result["YearsAtCompany"].clip(lower=1)
    total_exp = result["TotalWorkingYears"].clip(lower=1)
    num_cos = result["NumCompaniesWorked"].clip(lower=1)

    result["promotion_stagnation_ratio"] = (
        result["YearsSinceLastPromotion"] / years_at_co
    )
    result["role_tenure_share"] = result["YearsInCurrentRole"] / years_at_co
    result["manager_tenure_share"] = result["YearsWithCurrManager"] / years_at_co

    role_medians = result["JobRole"].map(state.role_income_medians)
    result["income_vs_role_median"] = result["MonthlyIncome"] / role_medians

    result["compensation_per_experience"] = result["MonthlyIncome"] / total_exp
    result["avg_satisfaction_score"] = result[sat_cols].mean(axis=1)

    result["burnout_risk_flag"] = (
        (result["OverTime"] == "Yes") & (result["WorkLifeBalance"] <= wlb_threshold)
    ).astype(int)

    result["job_stability_index"] = result["TotalWorkingYears"] / num_cos

    return result


def _build_importance_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: ProjectConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode nominal columns for tree-based importance evaluation."""
    nominal = [c for c in config.features.nominal if c in feature_cols]
    numeric = [c for c in feature_cols if c not in nominal]

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    if nominal:
        encoded = encoder.fit_transform(df[nominal])
        encoded_names = list(encoder.get_feature_names_out(nominal))
        encoded_df = pd.DataFrame(encoded, columns=encoded_names)
        matrix = pd.concat([df[numeric].reset_index(drop=True), encoded_df], axis=1)
        all_names = numeric + encoded_names
    else:
        matrix = df[numeric].reset_index(drop=True)
        all_names = numeric

    matrix.columns = all_names
    return matrix, all_names


def evaluate_feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Rank features by Random Forest importance (reproducible random seed)."""
    cfg = config or load_config()
    matrix, names = _build_importance_matrix(df, feature_cols, cfg)

    target = (df[cfg.target_column] == cfg.positive_class).astype(int).values
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=cfg.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(matrix, target)

    importance = pd.DataFrame(
        {"feature": names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance["importance"] = importance["importance"].round(6)
    return importance.reset_index(drop=True)


def find_redundant_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.92,
) -> pd.DataFrame:
    """Return feature pairs with |Pearson r| at or above *threshold*."""
    numeric_cols = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    corr = df[numeric_cols].corr().abs()
    pairs: list[dict[str, Any]] = []

    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            r = corr.loc[col_a, col_b]
            if r >= threshold:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(r, 4)})

    return pd.DataFrame(pairs)


def resolve_redundancy(
    pairs: pd.DataFrame,
    importance: pd.DataFrame,
) -> tuple[list[str], list[RedundancyDecision]]:
    """Drop the lower-importance member of each redundant pair."""
    importance_map = importance.set_index("feature")["importance"].to_dict()
    to_drop: set[str] = set()
    decisions: list[RedundancyDecision] = []

    for _, row in pairs.iterrows():
        col_a, col_b, r = row["feature_a"], row["feature_b"], row["correlation"]
        if col_a in to_drop or col_b in to_drop:
            continue

        imp_a = importance_map.get(col_a, 0.0)
        imp_b = importance_map.get(col_b, 0.0)

        if imp_a == imp_b:
            continue

        if imp_a < imp_b:
            dropped, kept, imp_d, imp_k = col_a, col_b, imp_a, imp_b
        else:
            dropped, kept, imp_d, imp_k = col_b, col_a, imp_b, imp_a

        to_drop.add(dropped)
        decisions.append(
            RedundancyDecision(
                dropped_feature=dropped,
                kept_feature=kept,
                correlation=float(r),
                dropped_importance=float(imp_d),
                kept_importance=float(imp_k),
                reason=(
                    f"|r|={r:.3f} with '{kept}' "
                    f"(importance {imp_k:.4f} vs {imp_d:.4f})"
                ),
            )
        )

    return sorted(to_drop), decisions


def _stratified_dataframe_split(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold-out split on full rows; matches modeling train/test indices."""
    from sklearn.model_selection import train_test_split

    y = (df[config.target_column] == config.positive_class).astype(int)
    return train_test_split(
        df,
        test_size=config.modeling.test_size,
        random_state=config.random_state,
        stratify=y,
    )


def run_feature_engineering_pipeline(
    df: pd.DataFrame,
    config: ProjectConfig | None = None,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    """Engineer features, evaluate importance, resolve redundancy, and save.

    Learned statistics (role income medians, feature importance, redundancy
    decisions) are fit on the training split only; transforms are applied to
    both training and hold-out rows before persisting the featured dataset.
    """
    cfg = config or load_config()
    base_features = cfg.model_feature_columns

    train_df, test_df = _stratified_dataframe_split(df, cfg)

    state = fit_feature_engineering_state(train_df, cfg)
    featured_train = apply_engineered_features(train_df, state, cfg)
    featured_test = apply_engineered_features(test_df, state, cfg)
    featured = pd.concat([featured_train, featured_test]).sort_index()
    engineered_cols = state.engineered_columns
    all_model_features = base_features + engineered_cols

    importance = evaluate_feature_importance(featured_train, all_model_features, cfg)
    pairs = find_redundant_pairs(
        featured_train,
        all_model_features,
        threshold=cfg.feature_engineering.correlation_threshold,
    )
    dropped, decisions = resolve_redundancy(pairs, importance)
    selected = [c for c in all_model_features if c not in dropped]

    report = FeatureEngineeringReport(
        input_shape=df.shape,
        output_shape=featured.shape,
        engineered_columns=engineered_cols,
        selected_features=selected,
        dropped_features=dropped,
        redundancy_decisions=decisions,
        importance_ranking=importance,
    )

    if save_artifacts:
        report.output_paths = save_feature_engineering_artifacts(
            featured, state, selected, dropped, decisions, config=cfg
        )

    return featured, report


def save_feature_engineering_artifacts(
    featured: pd.DataFrame,
    state: FeatureEngineeringState,
    selected_features: list[str],
    dropped_features: list[str],
    decisions: list[RedundancyDecision],
    config: ProjectConfig | None = None,
    output_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, str]:
    """Persist featured dataset, fit state, and selected feature list."""
    import json

    cfg = config or load_config()
    out_dir = output_dir or DATA_PROCESSED_DIR
    mdir = models_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    featured_path = out_dir / cfg.feature_engineered_filename
    state_path = mdir / "feature_engineering_state.joblib"
    selected_path = mdir / "selected_features.json"

    featured.to_parquet(featured_path, index=False)
    joblib.dump(state, state_path)

    payload = {
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "drop_justifications": [
            {
                "dropped": d.dropped_feature,
                "kept": d.kept_feature,
                "correlation": d.correlation,
                "reason": d.reason,
            }
            for d in decisions
        ],
        "random_state": cfg.random_state,
    }
    selected_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "featured": str(featured_path),
        "state": str(state_path),
        "selected_features": str(selected_path),
    }


def load_feature_engineered_data(
    config: ProjectConfig | None = None,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the dataset with engineered features."""
    cfg = config or load_config()
    path = (input_dir or DATA_PROCESSED_DIR) / cfg.feature_engineered_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Feature-engineered data not found at {path}. "
            "Run the feature engineering notebook first."
        )
    return pd.read_parquet(path)
