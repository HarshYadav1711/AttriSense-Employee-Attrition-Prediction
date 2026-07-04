"""Attrition Prediction page."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.components.layout import (
    load_theme,
    page_footer,
    page_header,
    render_risk_badge,
    render_validation_messages,
    section_title,
)
from app.services.data_service import get_categorical_options
from app.services.prediction_service import (
    default_employee_record,
    feature_frame_cache_key,
    get_shap_explanation,
    input_schema_dataframe,
    run_batch_prediction,
    run_single_prediction,
    validate_batch,
    validate_single,
)
from attrisense.inference import shap_contributions_table


def _render_confidence_gauge(probability: float, confidence: float) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Probability", f"{probability * 100:.1f}%")
    col2.metric("Confidence Score", f"{confidence * 100:.1f}%", help="Distance from uncertain (50%)")
    prediction = "Yes" if probability >= 0.5 else "No"
    col3.metric("Predicted Outcome", prediction)


def _shap_bar_plot(contributions: pd.DataFrame) -> None:
    plot_df = contributions.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#dc2626" if v >= 0 else "#059669" for v in plot_df["shap_value"]]
    ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_xlabel("SHAP value (impact on attrition log-odds)")
    ax.set_title("Top Feature Contributions")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _single_employee_form(options: dict) -> None:
    defaults = default_employee_record(options)

    section_title("Employee Profile")

    with st.form("single_prediction_form", clear_on_submit=False):
        st.number_input("Employee Number (optional)", min_value=1, step=1, key="emp_id")

        st.markdown("**Demographics & Role**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = c1.number_input("Age", 18, 70, defaults["Age"])
            gender = c1.selectbox("Gender", options["Gender"], index=options["Gender"].index(defaults["Gender"]))
            marital = c1.selectbox(
                "Marital Status",
                options["MaritalStatus"],
                index=options["MaritalStatus"].index(defaults["MaritalStatus"]),
            )
        with c2:
            department = c2.selectbox(
                "Department",
                options["Department"],
                index=options["Department"].index(defaults["Department"]),
            )
            job_role = c2.selectbox(
                "Job Role",
                options["JobRole"],
                index=options["JobRole"].index(defaults["JobRole"]),
            )
            education = c2.number_input("Education (1–5)", 1, 5, defaults["Education"])
        with c3:
            education_field = c3.selectbox(
                "Education Field",
                options["EducationField"],
                index=options["EducationField"].index(defaults["EducationField"]),
            )
            business_travel = c3.selectbox(
                "Business Travel",
                options["BusinessTravel"],
                index=options["BusinessTravel"].index(defaults["BusinessTravel"]),
            )
            overtime = c3.selectbox("OverTime", ["No", "Yes"], index=0 if defaults["OverTime"] == "No" else 1)

        st.markdown("**Compensation**")
        c4, c5, c6 = st.columns(3)
        with c4:
            monthly_income = c4.number_input("Monthly Income", min_value=1000, max_value=25000, value=defaults["MonthlyIncome"])
            daily_rate = c4.number_input("Daily Rate", min_value=100, value=defaults["DailyRate"])
        with c5:
            hourly_rate = c5.number_input("Hourly Rate", min_value=10, value=defaults["HourlyRate"])
            monthly_rate = c5.number_input("Monthly Rate", min_value=1000, value=defaults["MonthlyRate"])
        with c6:
            percent_hike = c6.number_input("Percent Salary Hike", min_value=0, max_value=50, value=defaults["PercentSalaryHike"])
            stock_level = c6.number_input("Stock Option Level (0–3)", 0, 3, defaults["StockOptionLevel"])

        st.markdown("**Tenure & Experience**")
        c7, c8, c9 = st.columns(3)
        with c7:
            total_years = c7.number_input("Total Working Years", 0, 50, defaults["TotalWorkingYears"])
            years_company = c7.number_input("Years at Company", 0, 50, defaults["YearsAtCompany"])
            num_companies = c7.number_input("Number of Companies Worked", 0, 15, defaults["NumCompaniesWorked"])
        with c8:
            years_role = c8.number_input("Years in Current Role", 0, 50, defaults["YearsInCurrentRole"])
            years_promotion = c8.number_input("Years Since Last Promotion", 0, 50, defaults["YearsSinceLastPromotion"])
            years_manager = c8.number_input("Years with Current Manager", 0, 50, defaults["YearsWithCurrManager"])
        with c9:
            distance = c9.number_input("Distance From Home", 1, 30, defaults["DistanceFromHome"])
            training = c9.number_input("Training Times Last Year", 0, 10, defaults["TrainingTimesLastYear"])
            performance = c9.selectbox("Performance Rating", options["PerformanceRating"], index=0)

        st.markdown("**Engagement & Satisfaction (1 = lowest, 4 = highest)**")
        c10, c11, c12, c13 = st.columns(4)
        job_sat = c10.slider("Job Satisfaction", 1, 4, defaults["JobSatisfaction"])
        env_sat = c11.slider("Environment Satisfaction", 1, 4, defaults["EnvironmentSatisfaction"])
        rel_sat = c12.slider("Relationship Satisfaction", 1, 4, defaults["RelationshipSatisfaction"])
        wlb = c13.slider("Work-Life Balance", 1, 4, defaults["WorkLifeBalance"])
        job_involvement = st.slider("Job Involvement", 1, 4, defaults["JobInvolvement"])

        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if not submitted:
        return

    record = {
        "EmployeeNumber": st.session_state.get("emp_id", 1),
        "Age": age,
        "Gender": gender,
        "MaritalStatus": marital,
        "Department": department,
        "JobRole": job_role,
        "Education": education,
        "EducationField": education_field,
        "BusinessTravel": business_travel,
        "OverTime": overtime,
        "MonthlyIncome": monthly_income,
        "DailyRate": daily_rate,
        "HourlyRate": hourly_rate,
        "MonthlyRate": monthly_rate,
        "PercentSalaryHike": percent_hike,
        "StockOptionLevel": stock_level,
        "TotalWorkingYears": total_years,
        "YearsAtCompany": years_company,
        "NumCompaniesWorked": num_companies,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager,
        "DistanceFromHome": distance,
        "TrainingTimesLastYear": training,
        "PerformanceRating": performance,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "RelationshipSatisfaction": rel_sat,
        "WorkLifeBalance": wlb,
        "JobInvolvement": job_involvement,
    }

    if render_validation_messages(validate_single(record)):
        return

    result, output_df = run_single_prediction(record, threshold)
    prob = float(result.probabilities[0])
    conf = float(result.confidence_scores[0])
    tier = result.risk_tiers[0]

    section_title("Prediction Result")
    st.markdown(render_risk_badge(tier), unsafe_allow_html=True)
    st.write("")
    _render_confidence_gauge(prob, conf)

    st.caption(
        "Confidence reflects how far the probability is from 50% (uncertain). "
        "It does not mean the prediction is guaranteed correct."
    )

    section_title("SHAP Explanation")
    explanation = get_shap_explanation(feature_frame_cache_key(result.feature_frame), row_index=0)
    contributions = shap_contributions_table(explanation, top_n=12)
    _shap_bar_plot(contributions)
    st.dataframe(contributions, use_container_width=True, hide_index=True)

    section_title("Export")
    st.download_button(
        "Download prediction (CSV)",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name=f"attrisense_prediction_{record['EmployeeNumber']}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _batch_upload_form() -> None:
    section_title("Batch Upload")
    st.markdown("Upload a CSV containing the required employee input columns.")

    with st.expander("Required column schema"):
        st.dataframe(input_schema_dataframe(), use_container_width=True, hide_index=True)

    from app.services.data_service import get_raw_data
    template = get_raw_data()[
        [c for c in input_schema_dataframe()["RequiredColumn"].tolist()]
    ].head(3)
    st.download_button(
        "Download sample template (CSV)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="attrisense_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Employee CSV", type=["csv"])
    threshold = st.slider("Decision threshold (batch)", 0.1, 0.9, 0.5, 0.05, key="batch_threshold")

    if uploaded is None:
        return

    try:
        upload_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if render_validation_messages(validate_batch(upload_df)):
        return

    output_df = run_batch_prediction(upload_df, threshold)
    section_title(f"Batch Results ({len(output_df):,} employees)")
    st.dataframe(output_df, use_container_width=True, hide_index=True)

    summary = output_df["RiskTier"].value_counts().reset_index()
    summary.columns = ["Risk Tier", "Count"]
    st.bar_chart(summary, x="Risk Tier", y="Count")

    st.download_button(
        "Download batch predictions (CSV)",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="attrisense_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render() -> None:
    load_theme()
    page_header(
        "Attrition Prediction",
        "Score attrition risk for individual employees or workforce batches. "
        "Includes probability, confidence, and SHAP-based explanations.",
    )

    options = get_categorical_options()
    tab_single, tab_batch = st.tabs(["Single Employee", "Batch Upload"])

    with tab_single:
        _single_employee_form(options)
    with tab_batch:
        _batch_upload_form()

    page_footer()
