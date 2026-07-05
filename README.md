# AttriSense

AttriSense is an end-to-end employee attrition analytics platform that helps HR teams identify employees who are at risk of leaving before attrition actually happens.

The project combines exploratory analytics, business-driven feature engineering, machine learning, model explainability, and an interactive prediction interface into a single reproducible workflow.

Built on the [IBM HR Analytics Attrition dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (1,470 employees, ~16% attrition rate).

Rather than focusing only on predictive accuracy, the system is designed to answer an equally important question:

> *Why is the employee considered high risk?*

Every prediction is accompanied by interpretable explanations, confidence estimates, and business-friendly insights that HR teams can use to support retention decisions.

---

## Why I Built This

Employee attrition is one of the most expensive challenges organizations face.

Replacing experienced employees requires recruitment, onboarding, training, and months of productivity recovery. Most existing demonstrations simply train a model and report an accuracy score.

I wanted to build something closer to a real internal analytics product.

That meant building an entire pipeline instead of a notebook:

- reproducible preprocessing
- feature engineering
- multiple candidate models
- model comparison
- explainability
- validation
- deployment
- documentation
- reusable codebase

The result is AttriSense.

---

# Features

### Workforce Analytics

- Interactive exploratory data analysis
- Distribution analysis
- Correlation analysis
- Attrition segmentation
- Department-level insights
- Workforce statistics
- Business trend visualizations

---

### Feature Engineering

Business-inspired engineered features including:

- Promotion stagnation ratio
- Role tenure share
- Manager tenure share
- Compensation relative to role median
- Experience-adjusted compensation
- Composite satisfaction score
- Burnout risk indicator
- Job stability index

Learned statistics (role income medians, feature importance, redundancy decisions) are fit on the training split only; transforms are then applied to all rows.

Feature redundancy is detected and removed using correlation analysis combined with feature importance, using training data only.

---

### Machine Learning Pipeline

The training pipeline evaluates multiple algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The pipeline includes:

- Stratified train/test split (80/20)
- 5-fold stratified cross-validation (ROC-AUC scoring)
- Hyperparameter optimization
- Model persistence
- Automatic comparison
- Reproducible training

---

### Model Evaluation

Models are evaluated using multiple complementary metrics instead of relying only on accuracy.

Metrics include:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score
- PR-AUC
- Balanced Accuracy
- Matthews Correlation Coefficient
- Brier Score

The evaluation stage also generates:

- ROC curves
- Precision-Recall curves
- Calibration curves
- Confusion matrices
- Feature importance rankings
- Overfitting analysis
- Threshold optimization

---

### Explainable Predictions

Every prediction includes:

- Probability of attrition
- Risk category
- Confidence score
- SHAP explanations
- Top contributing features

Instead of returning only **Yes** or **No**, AttriSense explains *why* the prediction was made.

---

### Prediction Interface

The Streamlit application supports:

- Individual employee prediction
- Batch CSV prediction
- Automatic validation
- Decision threshold defaulting to the value saved during evaluation (user-adjustable)
- Downloadable prediction reports
- Interactive visualizations
- SHAP interpretation

---

# Architecture

```
Raw Dataset
      │
      ▼
Data Validation
      │
      ▼
Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Prediction Engine
      │
      ▼
Interactive Dashboard
```

---

# Project Structure

```
app/
    Streamlit application

configs/
    Project configuration

data/
    Raw and processed datasets

docs/
    Technical documentation

models/
    Saved pipelines
    Evaluation reports
    Model artifacts

notebooks/
    Research and analysis

reports/
    Generated figures

scripts/
    Automation utilities

src/
    Core package

tests/
    Smoke tests
```

---

# Technology Stack

### Machine Learning

- Scikit-learn
- XGBoost
- SHAP

### Data Processing

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

### Application

- Streamlit

### Utilities

- Joblib
- YAML Configuration

---

# Reproducibility

The entire workflow is deterministic.

Configuration lives in `configs/config.yaml` (`random_state: 42`, feature types, modeling defaults). Using the same dataset and configuration reproduces the same preprocessing steps, engineered features, trained models, evaluation metrics, and prediction outputs.

Artifacts generated during the pipeline include:

- trained models
- selected features
- feature engineering state
- evaluation reports (including optimal threshold and overfitting summary)
- comparison metrics
- visualizations

---

# Running the Project

**Requirements:** Python 3.11+, raw CSV at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` ([Kaggle download](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset))

Clone the repository

```bash
git clone https://github.com/HarshYadav1711/AttriSense-Employee-Attrition-Prediction.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Run the complete pipeline

```bash
python scripts/run_pipeline.py
```

Launch the application

```bash
python scripts/run_app.py
```

Equivalent:

```bash
streamlit run app/main.py
```

Further detail: [docs/WORKFLOW.md](docs/WORKFLOW.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

# Current Best Model

After evaluating multiple algorithms, Logistic Regression provided the strongest balance between predictive performance and generalization on the hold-out dataset.

Highlights on the hold-out test set (default 0.5 decision threshold):

- ROC-AUC ≈ 0.82
- Recall ≈ 0.70 for identifying potential attrition
- Precision ≈ 0.38
- Better generalization than tree-based models (moderate vs. severe overfitting)
- Calibration diagnostics available in evaluation reports
- Interpretable coefficients

The optimal decision threshold (~0.78, F1-maximized on a validation split) is saved in `models/evaluation_results.json` and used as the Streamlit default.

The system is intended for employee risk prioritization and HR decision support—not as a replacement for human judgment.

---

# Engineering Principles

While building AttriSense, the emphasis was on writing software that remains maintainable beyond experimentation.

Key design decisions include:

- modular package structure
- configuration-driven pipeline
- reusable components
- reproducible experiments
- artifact persistence
- clear separation between training and inference
- validation before prediction
- explainability built into inference
- production-style project organization

---

# Future Improvements

Potential future enhancements include:

- REST API deployment
- Dockerized infrastructure
- MLflow experiment tracking
- CI/CD automation
- Model monitoring
- Drift detection
- Fairness analysis
- Incremental retraining
- Cloud deployment

---

# License

MIT License