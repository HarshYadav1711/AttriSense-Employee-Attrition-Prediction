"""Plotly chart builders for Streamlit pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#1e3a5f"
ACCENT = "#2563eb"
MUTED = "#64748b"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Segoe UI, sans-serif", color=MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def attrition_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Overall attrition count bar chart."""
    counts = df["Attrition"].value_counts().reset_index()
    counts.columns = ["Attrition", "Count"]
    fig = px.bar(
        counts,
        x="Attrition",
        y="Count",
        color="Attrition",
        color_discrete_map={"Yes": "#dc2626", "No": "#059669"},
        text="Count",
    )
    fig.update_traces(textposition="outside")
    return _base_layout(fig, "Workforce Attrition Overview")


def attrition_by_category_chart(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Attrition rate by a categorical column."""
    summary = (
        df.groupby(column)["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
        .sort_values("AttritionRate", ascending=False)
    )
    fig = px.bar(
        summary,
        x=column,
        y="AttritionRate",
        text=summary["AttritionRate"].round(1),
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_yaxes(title="Attrition Rate (%)")
    return _base_layout(fig, title)


def satisfaction_chart(df: pd.DataFrame) -> go.Figure:
    """Average satisfaction scores by attrition status."""
    cols = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    melted = df.melt(
        id_vars=["Attrition"],
        value_vars=cols,
        var_name="Dimension",
        value_name="Score",
    )
    summary = melted.groupby(["Attrition", "Dimension"], as_index=False)["Score"].mean()
    fig = px.bar(
        summary,
        x="Dimension",
        y="Score",
        color="Attrition",
        barmode="group",
        color_discrete_map={"Yes": "#dc2626", "No": "#059669"},
    )
    fig.update_yaxes(range=[0, 4.5], title="Average Score (1–4)")
    return _base_layout(fig, "Satisfaction Scores by Attrition Status")


def metrics_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of model metrics."""
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    melted = comparison.melt(id_vars="model", value_vars=metric_cols, var_name="Metric", value_name="Score")
    melted["model"] = melted["model"].str.replace("_", " ").str.title()
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="model",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_yaxes(range=[0, 1.05])
    return _base_layout(fig, "Model Comparison — Test Set Metrics")


def roc_curve_chart(curves: list[tuple[str, list, list, float]]) -> go.Figure:
    """Overlay ROC curves for multiple models."""
    fig = go.Figure()
    for name, fpr, tpr, auc in curves:
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8"),
            name="Random",
        )
    )
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _base_layout(fig, "ROC Curves — Test Set")


def confusion_matrix_heatmap(matrix: list[list[int]], title: str) -> go.Figure:
    """Confusion matrix heatmap."""
    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted Stay", "Predicted Leave"],
        y=["Actual Stay", "Actual Leave"],
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    return _base_layout(fig, title)


def feature_importance_chart(importance_df: pd.DataFrame, title: str, value_col: str) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    plot_df = importance_df.head(12).iloc[::-1]
    fig = px.bar(
        plot_df,
        x=value_col,
        y="feature",
        orientation="h",
        color_discrete_sequence=[NAVY],
    )
    fig.update_xaxes(title=value_col.replace("_", " ").title())
    return _base_layout(fig, title)
