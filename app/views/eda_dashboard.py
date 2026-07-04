"""EDA Dashboard page."""

import streamlit as st

from app.components.charts import (
    attrition_bar_chart,
    attrition_by_category_chart,
    satisfaction_chart,
)
from app.components.layout import (
    load_theme,
    page_footer,
    page_header,
    render_metric_row,
    section_title,
)
from app.services.data_service import get_raw_data


def render() -> None:
    load_theme()
    page_header(
        "EDA Dashboard",
        "Exploratory analysis of attrition drivers across the workforce population.",
    )

    df = get_raw_data()
    leavers = (df["Attrition"] == "Yes").sum()
    stayers = (df["Attrition"] == "No").sum()
    ot_rate = df.loc[df["OverTime"] == "Yes", "Attrition"].eq("Yes").mean() * 100

    render_metric_row(
        [
            ("Attrition Count", f"{leavers:,}", f"{leavers / len(df) * 100:.1f}%"),
            ("Retained", f"{stayers:,}", None),
            ("Overtime Attrition", f"{ot_rate:.1f}%", "When OverTime = Yes"),
            ("Avg Tenure (Years)", f"{df['YearsAtCompany'].mean():.1f}", None),
        ]
    )

    section_title("Population Overview")
    st.plotly_chart(attrition_bar_chart(df), use_container_width=True)

    section_title("Segment Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Department", "Job Role", "Overtime", "Satisfaction"])

    with tab1:
        st.plotly_chart(
            attrition_by_category_chart(df, "Department", "Attrition Rate by Department"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            attrition_by_category_chart(df, "JobRole", "Attrition Rate by Job Role"),
            use_container_width=True,
        )
    with tab3:
        st.plotly_chart(
            attrition_by_category_chart(df, "OverTime", "Attrition Rate by Overtime Status"),
            use_container_width=True,
        )
    with tab4:
        st.plotly_chart(satisfaction_chart(df), use_container_width=True)

    section_title("Key Observations")
    st.markdown(
        """
        - **Class imbalance:** ~16% attrition — accuracy alone is a poor success metric.
        - **Overtime:** Employees working overtime show materially higher attrition.
        - **Satisfaction:** Lower job and work-life balance scores align with departures.
        - **Income & role:** Sales roles and below-median compensation correlate with turnover.
        """
    )

    page_footer()
