# AttriSense

Workforce attrition analytics and prediction for HR teams. AttriSense turns employee profile data into ranked retention risk, explanatory insights, and an interactive dashboard — all running locally on your machine.

Built on the [IBM HR Analytics Attrition dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (1,470 employees, ~16% attrition rate).

## What it does

- **Explores** workforce data with filters, summaries, and EDA charts
- **Trains** four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) with stratified cross-validation
- **Evaluates** models on hold-out metrics, confusion matrices, ROC curves, and feature importance
- **Predicts** attrition probability for individual employees or CSV batches, with SHAP explanations
- **Deploys** as a multi-page Streamlit app suitable for internal HR analytics

The production model (Logistic Regression) achieves test ROC-AUC ~0.82. It is intended for **risk ranking and triage**, not deterministic individual predictions.

## Quick start

**Requirements:** Python 3.11+, raw CSV placed at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Run the full pipeline (preprocessing → features → training → evaluation):

```bash
python scripts/run_pipeline.py
```

Launch the dashboard:

```bash
streamlit run app/main.py
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | End-to-end pipeline, notebooks, artifacts, reproducibility |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, module layout, data flow |
| [notebooks/README.md](notebooks/README.md) | Notebook order and output clearing |

Configuration lives in `configs/config.yaml` (paths, feature types, model hyperparameters, random seed).

## Development

```bash
pip install -e ".[dev]"
pytest
python scripts/run_pipeline.py
python scripts/clear_notebook_outputs.py   # before committing notebooks
ruff check src app scripts tests
```

## Project layout

```
configs/          YAML configuration
data/raw/         Source CSV (not committed)
data/processed/   Generated parquet files
src/attrisense/   Core Python package
notebooks/        Interactive analysis pipeline
models/           Trained pipelines and evaluation JSON
reports/figures/  EDA and evaluation plots
app/              Streamlit application
scripts/          Pipeline automation
```

## Tech stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · Plotly · Streamlit · Joblib

## License

MIT
