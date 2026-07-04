"""About page."""

import streamlit as st

from app.components.layout import load_theme, page_footer, page_header, section_title
from attrisense.utils.paths import PROJECT_ROOT


def render() -> None:
    load_theme()
    page_header(
        "About AttriSense",
        "Documentation for the internal HR attrition analytics platform.",
    )

    section_title("Purpose")
    st.markdown(
        """
        AttriSense helps HR teams **prioritize retention conversations** by estimating attrition
        probability from workforce data. It is designed as an internal analytics tool — not a
        replacement for manager judgment or employee dialogue.
        """
    )

    section_title("Architecture")
    st.markdown(
        f"""
        | Layer | Location |
        |-------|----------|
        | Configuration | `configs/config.yaml` |
        | Data pipeline | `src/attrisense/data/` |
        | Model training | `src/attrisense/models/` |
        | Inference & SHAP | `src/attrisense/inference.py` |
        | Streamlit app | `app/` |
        | Model artifacts | `models/best_model.joblib` |

        **Project root:** `{PROJECT_ROOT}`
        """
    )

    section_title("Model Summary")
    st.markdown(
        """
        - **Algorithm:** Logistic Regression (class-weight balanced)
        - **Selection criteria:** Highest test ROC-AUC, F1, and recall with smallest overfit gap
        - **Test ROC-AUC:** ~0.82 (moderate discriminative ability)
        - **Test precision:** ~39% on attrition class
        - **Intended use:** Risk ranking and triage, not individual certainty
        """
    )

    section_title("Running Locally")
    st.code(
        """python -m venv .venv
.venv\\Scripts\\activate        # Windows
pip install -r requirements.txt
pip install -e .
streamlit run app/main.py""",
        language="bash",
    )

    section_title("Privacy & Dependencies")
    st.markdown(
        """
        - **No authentication** — suitable for trusted internal networks only
        - **No cloud services** — all data and models remain on this machine
        - **No external API calls** during prediction or analysis
        - **Open-source stack:** Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit, Plotly
        """
    )

    section_title("Dataset")
    st.markdown(
        """
        IBM HR Analytics Employee Attrition dataset (1,470 records).
        Target: `Attrition` (Yes/No), ~16% positive class.

        Source: [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
        """
    )

    page_footer()
