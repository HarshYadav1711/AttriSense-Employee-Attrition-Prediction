"""Model Insights page."""

import pandas as pd
import streamlit as st

from app.components.charts import (
    confusion_matrix_heatmap,
    feature_importance_chart,
    metrics_comparison_chart,
)
from app.components.layout import (
    load_theme,
    page_footer,
    page_header,
    render_metric_row,
    section_title,
)
from app.services.data_service import figures_dir, get_evaluation_results, get_training_results


def render() -> None:
    load_theme()
    page_header(
        "Model Insights",
        "Evaluation results for the production Logistic Regression model and comparison benchmarks.",
    )

    eval_results = get_evaluation_results()
    training = get_training_results()

    if not eval_results or not training:
        st.warning("Model artifacts not found. Run training and evaluation notebooks first.")
        page_footer()
        return

    best_name = eval_results["best_model"].replace("_", " ").title()
    best_metrics = eval_results["metrics_comparison"][0]

    render_metric_row(
        [
            ("Selected Model", best_name, None),
            ("Test ROC-AUC", f"{best_metrics['roc_auc']:.3f}", "Primary metric"),
            ("Test F1", f"{best_metrics['f1']:.3f}", None),
            ("Test Recall", f"{best_metrics['recall']:.1%}", "Attrition class"),
        ]
    )

    section_title("Selection Rationale")
    st.markdown(eval_results["selection_rationale"])

    section_title("Metrics Comparison")
    comparison = pd.DataFrame(eval_results["metrics_comparison"])
    st.plotly_chart(metrics_comparison_chart(comparison), use_container_width=True)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    section_title("Generalization")
    if "overfitting_report" in eval_results:
        st.dataframe(pd.DataFrame(eval_results["overfitting_report"]), use_container_width=True, hide_index=True)
    else:
        gen_rows = []
        for model in training["models"]:
            gen_rows.append(
                {
                    "Model": model["name"].replace("_", " ").title(),
                    "Train AUC": round(model["train_metrics"]["roc_auc"], 4),
                    "Test AUC": round(model["test_metrics"]["roc_auc"], 4),
                    "AUC Gap": round(model["train_metrics"]["roc_auc"] - model["test_metrics"]["roc_auc"], 4),
                    "Test Precision": round(model["test_metrics"]["precision"], 4),
                }
            )
        st.dataframe(pd.DataFrame(gen_rows), use_container_width=True, hide_index=True)

    section_title("Confusion Matrix — Best Model")
    cm = eval_results["confusion_matrices"][eval_results["best_model"]]
    st.plotly_chart(
        confusion_matrix_heatmap(cm, f"{best_name} — Test Set Confusion Matrix"),
        use_container_width=True,
    )

    section_title("ROC Curves")
    fig_path = figures_dir() / "roc_curves.png"
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("ROC curve figure not found. Re-run evaluation to generate plots.")

    section_title("Precision-Recall Curves")
    pr_path = figures_dir() / "precision_recall_curves.png"
    if pr_path.exists():
        st.image(str(pr_path), use_container_width=True)

    section_title("Calibration")
    cal_path = figures_dir() / "calibration_curves.png"
    if cal_path.exists():
        st.image(str(cal_path), use_container_width=True)

    if "optimal_threshold" in eval_results:
        section_title("Decision Threshold")
        st.markdown(
            f"Optimal threshold (validation): **{eval_results['optimal_threshold']:.2f}** "
            f"via `{eval_results.get('threshold_optimization', {}).get('method', 'f1')}` optimization."
        )
        if eval_results.get("threshold_analysis"):
            st.dataframe(
                pd.DataFrame(eval_results["threshold_analysis"]),
                use_container_width=True,
                hide_index=True,
            )

    section_title("Feature Importance — Best Model")
    best_model_data = next(m for m in eval_results["models"] if m["name"] == eval_results["best_model"])
    importance_df = pd.DataFrame(best_model_data["top_features"])
    value_col = "abs_coefficient" if "abs_coefficient" in importance_df.columns else "importance"
    st.plotly_chart(
        feature_importance_chart(importance_df, f"Top Features — {best_name}", value_col),
        use_container_width=True,
    )
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

    section_title("Saved Evaluation Figures")
    fdir = figures_dir()
    cols = st.columns(2)
    for i, pattern in enumerate(
        ["confusion_matrices.png", "metrics_comparison.png", "precision_recall_curves.png", "calibration_curves.png"]
    ):
        path = fdir / pattern
        if path.exists():
            cols[i % 2].image(str(path), caption=pattern.replace("_", " ").replace(".png", "").title())

    page_footer()
