"""AttriSense — multi-page Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched via `streamlit run app/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.pages import about, dataset_explorer, eda_dashboard, home, model_insights, prediction

st.set_page_config(
    page_title="AttriSense · HR Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(home.render, title="Home", icon="🏠", default=True),
    st.Page(dataset_explorer.render, title="Dataset Explorer", icon="📋"),
    st.Page(eda_dashboard.render, title="EDA Dashboard", icon="📈"),
    st.Page(prediction.render, title="Prediction", icon="🎯"),
    st.Page(model_insights.render, title="Model Insights", icon="🔬"),
    st.Page(about.render, title="About", icon="ℹ️"),
]

with st.sidebar:
    st.markdown("### AttriSense")
    st.caption("Employee Attrition Analytics")
    st.divider()

pg = st.navigation(pages, position="sidebar")
pg.run()
