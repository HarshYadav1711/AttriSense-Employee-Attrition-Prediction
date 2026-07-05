# Workflow

This document describes how to run AttriSense from a fresh clone through to the Streamlit dashboard. The README covers quick start; this file covers the full pipeline in detail.

## Prerequisites

1. Python 3.11 or newer
2. Raw dataset at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`  
   Download from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
3. Virtual environment with dependencies installed (see README)

## One-command pipeline

Regenerates all processed data, trains four models, evaluates them, and saves figures:

```bash
python scripts/run_pipeline.py
```

Expected outputs:

| Output | Location |
|--------|----------|
| Cleaned / preprocessed / featured parquet | `data/processed/` |
| Feature list and FE state | `models/selected_features.json`, `models/feature_engineering_state.joblib` |
| Trained pipelines | `models/{logistic_regression,decision_tree,random_forest,xgboost}.joblib` |
| Training summary | `models/training_results.json` |
| Evaluation summary + best model | `models/evaluation_results.json`, `models/best_model.joblib` |
| Comparison plots | `reports/figures/` (ROC, PR, calibration, confusion matrices, metrics) |

## Step-by-step (Python API)

Each stage can be run independently from a notebook or script:

```python
from attrisense.config import load_config
from attrisense.data import (
    load_raw_data,
    run_preprocessing_pipeline,
    load_processed_data,
    run_feature_engineering_pipeline,
)
from attrisense.models import run_training_pipeline, run_evaluation_pipeline

config = load_config()

# 1. Preprocessing
raw = load_raw_data(config)
run_preprocessing_pipeline(raw, config=config)

# 2. Feature engineering
df = load_processed_data(config)
run_feature_engineering_pipeline(df, config=config, save_artifacts=True)
# Role medians, feature importance, and redundancy decisions are fit on the
# training split only; transforms are applied to all rows before saving parquet.

# 3. Training
train_report = run_training_pipeline(config)
print(train_report.comparison)

# 4. Evaluation
eval_report = run_evaluation_pipeline(config)
print(eval_report.best_model, eval_report.metrics_comparison)
```

## Notebook pipeline

Notebooks orchestrate the same functions with narrative context. Recommended order:

| Step | Notebook | Focus |
|------|----------|-------|
| 1 | `01_problem_understanding.ipynb` | Business problem and success criteria |
| 2 | `01_Data_Understanding.ipynb` / `02_dataset_understanding.ipynb` | Schema, distributions, data quality |
| 3 | `03_data_cleaning.ipynb` | Constant columns, target validation |
| 4 | `02_Data_Preprocessing.ipynb` | Encoding, duplicates, outlier inspection |
| 5 | `03_EDA.ipynb` | Attrition drivers and segment analysis |
| 6 | `04_Feature_Engineering.ipynb` | Derived features, importance, redundancy |
| 7 | `05_Model_Training.ipynb` | CV tuning, model comparison |
| 8 | `06_Model_Evaluation.ipynb` | Full metric suite, model selection |

Install the project kernel for notebooks:

```bash
python -m ipykernel install --user --name=attrisense --display-name="AttriSense (.venv)"
```

## Streamlit application

After the pipeline completes:

```bash
streamlit run app/main.py
```

The app loads `models/best_model.joblib` and featured data from `data/processed/`. If artifacts are missing, prediction and model insight pages will show warnings.

### Prediction input

Single-employee and batch modes require the 29 columns defined in `attrisense.inference.INPUT_COLUMNS`. A sample template is available from the **Prediction → Batch Upload** tab.

## Reproducibility

| Control | Value | Location |
|---------|-------|----------|
| Random seed | `42` | `configs/config.yaml` |
| Train/test split | 80/20 stratified | `modeling.test_size` |
| CV folds | 5 (stratified) | `modeling.cv_folds` |
| Tuning metric | ROC-AUC | `modeling.scoring` |
| Decision threshold | F1-max on validation split | `modeling.threshold_optimization` |

All stochastic steps (split, CV shuffle, RF/XGB/SHAP background sampling) derive from the configured seed. Re-running `scripts/run_pipeline.py` on the same raw data produces identical metrics (verified on Python 3.11+ with pinned `requirements.txt` versions).

**Do not hand-edit** generated parquet, joblib, or JSON artifacts — regenerate them instead.

## Configuration changes

Edit `configs/config.yaml`, then rerun from the affected stage:

| Change | Rerun from |
|--------|------------|
| `drop_columns`, paths | Preprocessing |
| `feature_engineering.*` | Feature engineering |
| `modeling.*` | Training |
| Feature type lists | Preprocessing + downstream |

## Troubleshooting

**`FileNotFoundError` for raw CSV** — Place the Kaggle file in `data/raw/`.

**`FileNotFoundError` for parquet or joblib** — Run `python scripts/run_pipeline.py`.

**Streamlit import errors** — Use the project virtualenv; run `pip install -r requirements.txt` inside it.

**SHAP warning about background samples** — Informational only; explanations use a 120-row background sample (see `DEFAULT_SHAP_BACKGROUND_SIZE` in `inference.py`).

## Inference-only usage

To score employees without the dashboard:

```python
import pandas as pd
from attrisense.inference import predict_attrition, build_prediction_dataframe

employees = pd.read_csv("my_team.csv")  # must include INPUT_COLUMNS
result = predict_attrition(employees)
print(build_prediction_dataframe(result))
```

See `docs/ARCHITECTURE.md` for module-level design detail.
