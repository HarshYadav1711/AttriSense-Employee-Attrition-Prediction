"""Home page — workforce attrition overview."""

import streamlit as st

from app.components.layout import (
    load_theme,
    page_footer,
    page_header,
    render_hero,
    render_metric_row,
)
from app.services.data_service import get_dataset_stats, get_evaluation_results, get_raw_data


def render() -> None:
    load_theme()
    render_hero(
        "AttriSense",
        "Internal HR analytics for employee attrition risk — explore workforce data, "
        "review model performance, and score retention priority.",
    )

    stats = get_dataset_stats()
    df = get_raw_data()
    attrition_rate = (df["Attrition"] == "Yes").mean() * 100
    eval_results = get_evaluation_results()
    best_auc = None
    if eval_results and eval_results.get("metrics_comparison"):
        best_auc = eval_results["metrics_comparison"][0]["roc_auc"]

    render_metric_row(
        [
            ("Employees", f"{stats['rows']:,}", None),
            ("Features", str(stats["columns"]), None),
            ("Attrition Rate", f"{attrition_rate:.1f}%", "Imbalanced"),
            (
                "Model ROC-AUC",
                f"{best_auc:.3f}" if best_auc is not None else "N/A",
                "Test set" if best_auc is not None else None,
            ),
        ]
    )

    st.markdown("---")
    page_header("Platform Overview", "Navigate using the sidebar to access each HR analytics module.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="as-card">
                <div class="as-card-title">Dataset Explorer</div>
                Browse the IBM HR workforce dataset — filter records, inspect profiles,
                and export subsets for offline review.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="as-card">
                <div class="as-card-title">EDA Dashboard</div>
                Visualize attrition patterns across department, role, satisfaction,
                overtime, and compensation dimensions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="as-card">
                <div class="as-card-title">Attrition Prediction</div>
                Score individual employees or upload a CSV batch. Each prediction includes
                probability, risk tier, confidence, and SHAP explanations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="as-card">
                <div class="as-card-title">Model Insights</div>
                Review evaluation metrics, ROC curves, confusion matrices, and feature
                importance for the selected Logistic Regression model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="as-card">
                <div class="as-card-title">Responsible Use</div>
                Predictions support <strong>risk ranking and triage</strong>, not deterministic
                outcomes. Test ROC-AUC is ~0.82 with ~39% precision — roughly 6 in 10 flagged
                employees will not leave.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if eval_results:
            best = eval_results["best_model"].replace("_", " ").title()
            st.info(f"**Production model:** {best} · Saved locally at `models/best_model.joblib`")

    page_footer()
