"""AttriSense — multi-page Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched via `streamlit run app/main.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402


def _home_page() -> None:
    from app.views import home

    home.render()


def _dataset_explorer_page() -> None:
    from app.views import dataset_explorer

    dataset_explorer.render()


def _eda_dashboard_page() -> None:
    from app.views import eda_dashboard

    eda_dashboard.render()


def _prediction_page() -> None:
    from app.views import prediction

    prediction.render()


def _model_insights_page() -> None:
    from app.views import model_insights

    model_insights.render()


def _about_page() -> None:
    from app.views import about

    about.render()


st.set_page_config(
    page_title="AttriSense · HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(_home_page, title="Home", icon="🏠", url_path="home", default=True),
    st.Page(_dataset_explorer_page, title="Dataset Explorer", icon="📋", url_path="dataset-explorer"),
    st.Page(_eda_dashboard_page, title="EDA Dashboard", icon="📈", url_path="eda-dashboard"),
    st.Page(_prediction_page, title="Prediction", icon="🎯", url_path="prediction"),
    st.Page(_model_insights_page, title="Model Insights", icon="🔬", url_path="model-insights"),
    st.Page(_about_page, title="About", icon="ℹ️", url_path="about"),
]

with st.sidebar:
    st.markdown("### AttriSense")
    st.caption("Employee Attrition Analytics")
    st.divider()

pg = st.navigation(pages, position="sidebar")
with st.spinner("Loading AttriSense..."):
    pg.run()
