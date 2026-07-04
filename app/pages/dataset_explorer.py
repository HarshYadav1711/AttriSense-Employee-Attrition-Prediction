"""Dataset Explorer page."""

import streamlit as st

from app.components.layout import load_theme, page_footer, page_header, render_metric_row, section_title
from app.services.data_service import get_config, get_raw_data


def render() -> None:
    load_theme()
    page_header(
        "Dataset Explorer",
        "Browse and filter the IBM HR Analytics employee dataset. All data stays on this machine.",
    )

    cfg = get_config()
    df = get_raw_data()

    render_metric_row(
        [
            ("Total Records", f"{len(df):,}", None),
            ("Departments", str(df["Department"].nunique()), None),
            ("Job Roles", str(df["JobRole"].nunique()), None),
            ("Missing Values", str(int(df.isna().sum().sum())), None),
        ]
    )

    section_title("Filters")

    col1, col2, col3 = st.columns(3)
    with col1:
        departments = st.multiselect(
            "Department",
            sorted(df["Department"].unique()),
            default=sorted(df["Department"].unique()),
        )
    with col2:
        attrition = st.multiselect("Attrition", ["Yes", "No"], default=["Yes", "No"])
    with col3:
        overtime = st.multiselect("OverTime", ["Yes", "No"], default=["Yes", "No"])

    col4, col5 = st.columns(2)
    with col4:
        age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()), (18, 60))
    with col5:
        income_range = st.slider(
            "Monthly income range",
            int(df["MonthlyIncome"].min()),
            int(df["MonthlyIncome"].max()),
            (int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())),
        )

    search = st.text_input("Search by Employee Number", placeholder="e.g. 102")

    filtered = df[
        df["Department"].isin(departments)
        & df["Attrition"].isin(attrition)
        & df["OverTime"].isin(overtime)
        & df["Age"].between(age_range[0], age_range[1])
        & df["MonthlyIncome"].between(income_range[0], income_range[1])
    ]

    if search.strip():
        filtered = filtered[filtered[cfg.id_column].astype(str).str.contains(search.strip())]

    section_title(f"Results ({len(filtered):,} employees)")

    display_cols = [
        cfg.id_column,
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "OverTime",
        "JobSatisfaction",
        "WorkLifeBalance",
        "Attrition",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        label="Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="attrisense_filtered_workforce.csv",
        mime="text/csv",
        use_container_width=True,
    )

    page_footer()
