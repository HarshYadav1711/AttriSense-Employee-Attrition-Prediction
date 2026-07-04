"""Model training with stratified split, CV tuning, and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from attrisense.config import ModelingConfig, ProjectConfig, load_config
from attrisense.models.pipelines import build_model_pipeline
from attrisense.utils.paths import DATA_PROCESSED_DIR, MODELS_DIR


@dataclass
class ModelTrainingResult:
    """Outcome of training a single classifier."""

    model_name: str
    best_params: dict[str, Any]
    cv_best_score: float
    cv_scoring: str
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model_path: str


@dataclass
class TrainingReport:
    """Aggregated training run summary."""

    n_train: int
    n_test: int
    feature_count: int
    random_state: int
    results: list[ModelTrainingResult] = field(default_factory=list)
    comparison: pd.DataFrame | None = None
    results_path: str = ""


def load_selected_features(models_dir: Path | None = None) -> list[str]:
    """Load the feature list saved during feature engineering."""
    path = (models_dir or MODELS_DIR) / "selected_features.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Selected features not found at {path}. Run feature engineering first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["selected_features"])


def prepare_training_data(
    config: ProjectConfig | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load featured dataset and return X, y, and selected feature names."""
    cfg = config or load_config()
    path = (data_dir or DATA_PROCESSED_DIR) / cfg.feature_engineered_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Feature-engineered data not found at {path}. "
            "Run feature engineering first."
        )

    df = pd.read_parquet(path)
    features = load_selected_features()
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing from dataset: {missing}")

    x = df[features].copy()
    y = (df[cfg.target_column] == cfg.positive_class).astype(int)
    return x, y, features


def stratified_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out split using project configuration."""
    cfg = config or load_config()
    mcfg = cfg.modeling
    return train_test_split(
        x,
        y,
        test_size=mcfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _evaluate_pipeline(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    y_pred = pipeline.predict(x)
    y_prob = pipeline.predict_proba(x)[:, 1]
    return _classification_metrics(y.values, y_pred, y_prob)


def get_model_specs(config: ProjectConfig | None = None) -> dict[str, dict[str, Any]]:
    """Return model definitions: pipeline factory inputs and tuning grids."""
    cfg = config or load_config()

    return {
        "logistic_regression": {
            "scale_numeric": True,
            "estimator": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=cfg.random_state,
            ),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "saga"],
            },
        },
        "decision_tree": {
            "scale_numeric": False,
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=cfg.random_state,
            ),
            "param_grid": {
                "classifier__max_depth": [3, 5, 8, 12, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__min_samples_leaf": [1, 5, 10],
            },
        },
        "random_forest": {
            "scale_numeric": False,
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=cfg.random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__max_features": ["sqrt", "log2"],
            },
        },
        "xgboost": {
            "scale_numeric": False,
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=cfg.random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
        },
    }


def train_single_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    config: ProjectConfig | None = None,
    models_dir: Path | None = None,
) -> ModelTrainingResult:
    """Tune one model with stratified CV and persist the best pipeline."""
    cfg = config or load_config()
    mcfg = cfg.modeling
    mdir = models_dir or MODELS_DIR
    mdir.mkdir(parents=True, exist_ok=True)

    specs = get_model_specs(cfg)[model_name]
    estimator = specs["estimator"]

    if model_name == "xgboost":
        neg = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        estimator.set_params(scale_pos_weight=neg / max(pos_count, 1))

    pipeline = build_model_pipeline(
        estimator,
        feature_names,
        config=cfg,
        scale_numeric=specs["scale_numeric"],
    )

    cv = StratifiedKFold(
        n_splits=mcfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=specs["param_grid"],
        scoring=mcfg.scoring,
        cv=cv,
        n_jobs=mcfg.n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    best_pipeline: Pipeline = search.best_estimator_
    train_metrics = _evaluate_pipeline(best_pipeline, x_train, y_train)
    test_metrics = _evaluate_pipeline(best_pipeline, x_test, y_test)

    model_path = mdir / f"{model_name}.joblib"
    joblib.dump(best_pipeline, model_path)

    return ModelTrainingResult(
        model_name=model_name,
        best_params=search.best_params_,
        cv_best_score=float(search.best_score_),
        cv_scoring=mcfg.scoring,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_path=str(model_path),
    )


def run_training_pipeline(
    config: ProjectConfig | None = None,
    models_dir: Path | None = None,
    model_names: list[str] | None = None,
) -> TrainingReport:
    """Train all configured models and save comparison results."""
    cfg = config or load_config()
    mdir = models_dir or MODELS_DIR

    x, y, features = prepare_training_data(cfg)
    x_train, x_test, y_train, y_test = stratified_train_test_split(x, y, cfg)

    names = model_names or list(get_model_specs(cfg).keys())
    results: list[ModelTrainingResult] = []

    for name in names:
        results.append(
            train_single_model(
                name,
                x_train,
                y_train,
                x_test,
                y_test,
                features,
                config=cfg,
                models_dir=mdir,
            )
        )

    comparison = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "cv_score": round(r.cv_best_score, 4),
                "test_accuracy": round(r.test_metrics["accuracy"], 4),
                "test_precision": round(r.test_metrics["precision"], 4),
                "test_recall": round(r.test_metrics["recall"], 4),
                "test_f1": round(r.test_metrics["f1"], 4),
                "test_roc_auc": round(r.test_metrics["roc_auc"], 4),
            }
            for r in results
        ]
    ).sort_values("test_roc_auc", ascending=False)

    results_path = mdir / "training_results.json"
    payload = {
        "random_state": cfg.random_state,
        "test_size": cfg.modeling.test_size,
        "cv_folds": cfg.modeling.cv_folds,
        "scoring": cfg.modeling.scoring,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "feature_count": len(features),
        "models": [
            {
                "name": r.model_name,
                "best_params": r.best_params,
                "cv_best_score": r.cv_best_score,
                "train_metrics": r.train_metrics,
                "test_metrics": r.test_metrics,
                "model_path": r.model_path,
            }
            for r in results
        ],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return TrainingReport(
        n_train=len(x_train),
        n_test=len(x_test),
        feature_count=len(features),
        random_state=cfg.random_state,
        results=results,
        comparison=comparison,
        results_path=str(results_path),
    )


def load_trained_model(model_name: str, models_dir: Path | None = None) -> Pipeline:
    """Load a persisted model pipeline."""
    path = (models_dir or MODELS_DIR) / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)
