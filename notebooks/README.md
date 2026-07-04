# Notebooks

Interactive walkthrough of the AttriSense pipeline. Core logic lives in `src/attrisense/`; notebooks call those functions and document decisions.

## Recommended order

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_problem_understanding.ipynb` | Business context and success criteria |
| 2 | `01_Data_Understanding.ipynb` | Schema and field definitions |
| 2 | `02_dataset_understanding.ipynb` | Distributions and data quality *(alternative to step 2)* |
| 3 | `03_data_cleaning.ipynb` | Drop constant columns, validate target |
| 4 | `02_Data_Preprocessing.ipynb` | Encoding, duplicates, outlier inspection |
| 5 | `03_EDA.ipynb` | Attrition drivers and segment analysis |
| 6 | `04_Feature_Engineering.ipynb` | Derived features, importance, redundancy |
| 7 | `05_Model_Training.ipynb` | CV tuning and model comparison |
| 8 | `06_Model_Evaluation.ipynb` | Full evaluation and model selection |

Steps 2 has two alternative notebooks covering similar ground — use one or both.

For a non-interactive run, use `python scripts/run_pipeline.py` instead.

## Before committing

Clear execution outputs (removes machine-specific paths from diffs):

```bash
python scripts/clear_notebook_outputs.py
```
