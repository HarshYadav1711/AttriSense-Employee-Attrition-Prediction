"""Cached data access for the Streamlit application."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from attrisense.config import load_config
from attrisense.data import load_raw_data, dataset_summary
from attrisense.data.feature_engineering import load_feature_engineered_data
from attrisense.inference import load_best_model, load_selected_features
from attrisense.utils.paths import MODELS_DIR, REPORTS_FIGURES_DIR


@st.cache_data(show_spinner=False)
def get_config():
    return load_config()


@st.cache_data(show_spinner="Loading workforce dataset…")
def get_raw_data() -> pd.DataFrame:
    return load_raw_data(get_config())


@st.cache_data(show_spinner=False)
def get_dataset_stats() -> dict:
    return dataset_summary(get_raw_data())


@st.cache_data(show_spinner=False)
def get_featured_data() -> pd.DataFrame | None:
    try:
        return load_feature_engineered_data(get_config())
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading prediction model…")
def get_model():
    return load_best_model()


@st.cache_data(show_spinner=False)
def get_selected_features() -> list[str]:
    return load_selected_features()


@st.cache_data(show_spinner=False)
def get_training_results() -> dict | None:
    path = MODELS_DIR / "training_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_evaluation_results() -> dict | None:
    path = MODELS_DIR / "evaluation_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_categorical_options() -> dict[str, list]:
    df = get_raw_data()
    cfg = get_config()
    options = {col: sorted(df[col].dropna().unique().tolist()) for col in cfg.features.nominal}
    options["PerformanceRating"] = sorted(df["PerformanceRating"].unique().tolist())
    return options


def figures_dir():
    return REPORTS_FIGURES_DIR
