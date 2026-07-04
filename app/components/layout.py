"""Layout helpers for the AttriSense Streamlit application."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from attrisense.inference import ValidationIssue

THEME_PATH = Path(__file__).resolve().parents[1] / "styles" / "theme.css"


def load_theme() -> None:
    """Inject custom CSS theme."""
    if THEME_PATH.exists():
        st.markdown(f"<style>{THEME_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Render a consistent page title block."""
    st.markdown(f'<p class="as-page-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="as-page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    """Render a hero banner for the home page."""
    st.markdown(
        f"""
        <div class="as-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of metrics. Each item is (label, value, delta)."""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def render_risk_badge(tier: str) -> str:
    """Return HTML for a risk tier badge."""
    css_class = {
        "High": "as-risk-high",
        "Elevated": "as-risk-elevated",
        "Moderate": "as-risk-moderate",
        "Low": "as-risk-low",
    }.get(tier, "as-risk-low")
    return f'<span class="{css_class}">{tier} Risk</span>'


def render_validation_messages(issues: list[ValidationIssue]) -> bool:
    """Display validation issues. Returns True if blocking errors exist."""
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    for issue in errors:
        st.error(f"**{issue.field}**: {issue.message}")
    for issue in warnings:
        st.warning(f"**{issue.field}**: {issue.message}")

    return len(errors) > 0


def page_footer() -> None:
    """Render a subtle page footer."""
    st.markdown(
        '<div class="as-footer">AttriSense · Internal HR Analytics · Runs locally · No cloud dependency</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    """Render a section heading."""
    st.markdown(f'<p class="as-section-title">{title}</p>', unsafe_allow_html=True)
