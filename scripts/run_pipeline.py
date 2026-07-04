#!/usr/bin/env python
"""Run the full AttriSense pipeline: preprocess -> features -> train -> evaluate.

Usage:
    python scripts/run_pipeline.py

Exits with code 0 on success. Prints a summary of key artifacts and test metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from attrisense.config import load_config
    from attrisense.data import (
        load_processed_data,
        load_raw_data,
        run_feature_engineering_pipeline,
        run_preprocessing_pipeline,
    )
    from attrisense.models import run_evaluation_pipeline, run_training_pipeline
    from attrisense.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_FIGURES_DIR

    config = load_config()
    print("AttriSense pipeline")
    print("=" * 40)

    print("\n[1/4] Preprocessing...")
    raw = load_raw_data(config)
    run_preprocessing_pipeline(raw, config=config)
    for name in (
        config.processed_filename,
        config.preprocessed_filename,
    ):
        path = DATA_PROCESSED_DIR / name
        assert path.exists(), f"Missing {path}"
        print(f"  OK  {path.name}")

    print("\n[2/4] Feature engineering...")
    df = load_processed_data(config)
    run_feature_engineering_pipeline(df, config=config, save_artifacts=True)
    featured = DATA_PROCESSED_DIR / config.feature_engineered_filename
    assert featured.exists(), f"Missing {featured}"
    assert (MODELS_DIR / "selected_features.json").exists()
    print(f"  OK  {featured.name}")

    print("\n[3/4] Training...")
    train_report = run_training_pipeline(config)
    assert train_report.comparison is not None
    print(train_report.comparison.to_string(index=False))
    print(f"  OK  {len(train_report.results)} models saved")

    print("\n[4/4] Evaluation...")
    eval_report = run_evaluation_pipeline(config)
    assert (MODELS_DIR / "best_model.joblib").exists()
    assert (MODELS_DIR / "evaluation_results.json").exists()
    print(f"  Best model: {eval_report.best_model}")
    print(eval_report.metrics_comparison.to_string(index=False))

    figures = list(REPORTS_FIGURES_DIR.glob("*.png"))
    print(f"\n  OK  {len(figures)} figures in reports/figures/")

    results = json.loads((MODELS_DIR / "evaluation_results.json").read_text())
    best_auc = results["metrics_comparison"][0]["roc_auc"]
    print(f"\nPipeline complete. Test ROC-AUC (best): {best_auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
