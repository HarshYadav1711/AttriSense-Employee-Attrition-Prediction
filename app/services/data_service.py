"""Cached data loaders for the Streamlit application."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from attrisense.config import load_config
from attrisense.data import dataset_summary, load_raw_data
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
def get_optimal_threshold() -> float:
    results = get_evaluation_results()
    if not results:
        return 0.5
    return float(results.get("optimal_threshold", 0.5))


@st.cache_data(show_spinner=False)
def get_categorical_options() -> dict[str, list]:
    df = get_raw_data()
    cfg = get_config()
    options = {col: sorted(df[col].dropna().unique().tolist()) for col in cfg.features.nominal}
    options["PerformanceRating"] = sorted(df["PerformanceRating"].unique().tolist())
    return options


def figures_dir():
    return REPORTS_FIGURES_DIR
