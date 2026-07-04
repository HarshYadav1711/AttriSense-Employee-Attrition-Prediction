# Architecture

AttriSense is organized as a **config-driven ML pipeline** with a separate **presentation layer**. Business logic lives in `src/attrisense/`; the Streamlit app in `app/` consumes that package without reimplementing training or inference.

## System overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        configs/config.yaml                       │
│         (paths, feature types, FE params, model settings)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐      ┌─────────────┐     ┌─────────────┐
   │   data/   │      │ src/        │     │   app/      │
   │  raw CSV  │─────▶│ attrisense  │────▶│ Streamlit   │
   │  parquet  │      │  package    │     │  dashboard  │
   └───────────┘      └──────┬──────┘     └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   models/   │
                      │  .joblib    │
                      │  .json      │
                      └─────────────┘
```

## Package structure (`src/attrisense/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Load and type `configs/config.yaml` into dataclasses |
| `utils/paths.py` | Resolve project root and standard directories |
| `data/loader.py` | Read raw CSV |
| `data/cleaning.py` | Drop constant columns, validate target |
| `data/preprocessing.py` | Dedup, encode, outlier inspection, persist parquet |
| `data/feature_engineering.py` | Derived HR features, importance ranking, redundancy removal |
| `data/features.py` | Feature registry metadata |
| `models/pipelines.py` | Sklearn `ColumnTransformer` + classifier pipelines |
| `models/training.py` | GridSearchCV tuning, metrics, model persistence |
| `models/evaluation.py` | Confusion matrices, ROC curves, model selection |
| `inference.py` | Validation, batch/single prediction, SHAP explanations |

## Data flow

### Training path

1. **Raw CSV** → `clean_raw_data()` removes zero-variance columns
2. **Preprocessing** → deduplication, nominal encoding artifacts, cleaned/preprocessed parquet
3. **Feature engineering** → 8 derived features (e.g. `job_stability_index`, `burnout_risk_flag`); drops redundant pairs (e.g. `JobLevel` vs `MonthlyIncome`)
4. **Training** → stratified 80/20 split; 5-fold CV with ROC-AUC scoring; four models tuned via `GridSearchCV`
5. **Evaluation** → test metrics, plots saved to `reports/figures/`, best model copied to `best_model.joblib`

### Inference path

1. User provides **29 base input columns** (see `inference.INPUT_COLUMNS`)
2. `apply_engineered_features()` adds derived columns using saved `feature_engineering_state.joblib`
3. Subset to **37 selected features** from `selected_features.json`
4. `best_model.joblib` pipeline (preprocessor + Logistic Regression) returns probability
5. SHAP `LinearExplainer` explains individual predictions

## Model pipeline design

Each classifier is a sklearn `Pipeline`:

```
[ColumnTransformer]  →  [Classifier]
  ├─ OneHotEncoder (nominal, drop first)
  └─ StandardScaler or passthrough (numeric)
```

Logistic Regression uses scaling; tree models do not. Class imbalance is handled via `class_weight="balanced"` (linear/tree) or `scale_pos_weight` (XGBoost).

## Application layer (`app/`)

| Directory | Role |
|-----------|------|
| `main.py` | Page routing via `st.navigation` |
| `pages/` | One module per screen (Home, Explorer, EDA, Prediction, Insights, About) |
| `services/` | `@st.cache_*` wrappers around package loaders and predictors |
| `components/` | Shared layout helpers and Plotly charts |
| `styles/theme.css` | Internal HR analytics visual theme |

The app never trains models at runtime. It loads persisted artifacts from `models/`.

## Configuration as single source of truth

`configs/config.yaml` drives:

- File names and paths (relative to repo root)
- Feature typing (ordinal / nominal / continuous) — controls encoding and scaling
- Feature engineering thresholds (burnout flag, correlation cutoff)
- Modeling defaults (test split, CV folds, scoring metric, `random_state: 42`)

Changing config without rerunning the pipeline will desynchronize artifacts.

## Artifact contract

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `data/processed/*.parquet` | Preprocessing, FE | Training, notebooks, app |
| `models/selected_features.json` | Feature engineering | Training, inference |
| `models/feature_engineering_state.joblib` | Feature engineering | Inference |
| `models/{model}.joblib` | Training | Evaluation, app |
| `models/best_model.joblib` | Evaluation | App inference |
| `models/training_results.json` | Training | Evaluation, app |
| `models/evaluation_results.json` | Evaluation | App |

Generated artifacts are gitignored except JSON metadata where noted; regenerate via `python scripts/run_pipeline.py`.

## Design decisions

**Why Logistic Regression for production?** Highest test ROC-AUC (0.816), best F1/recall, smallest train–test gap. Tree models overfit (train AUC ≈ 1.0, weak recall).

**Why separate inference module?** Keeps validation, feature engineering, and SHAP logic testable outside Streamlit.

**Why YAML config?** Notebooks and scripts share one definition of feature types and paths — avoids hard-coded column lists scattered across files.

**Why no auth or cloud?** Scoped as an internal analytics tool; all data and models remain on the local machine.
