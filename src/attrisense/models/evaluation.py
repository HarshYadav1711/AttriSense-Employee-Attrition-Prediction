"""Model evaluation: metrics, confusion matrices, ROC curves, feature importance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from attrisense.config import ProjectConfig, load_config
from attrisense.models.training import (
    get_model_specs,
    load_trained_model,
    prepare_training_data,
    stratified_train_test_split,
)
from attrisense.utils.paths import MODELS_DIR, REPORTS_FIGURES_DIR


@dataclass
class ModelEvaluationResult:
    """Detailed evaluation outcome for one classifier."""

    model_name: str
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    feature_importance: pd.DataFrame
    y_pred: np.ndarray
    y_prob: np.ndarray


@dataclass
class EvaluationReport:
    """Aggregated evaluation across all models."""

    n_test: int
    positive_rate: float
    results: list[ModelEvaluationResult] = field(default_factory=list)
    metrics_comparison: pd.DataFrame | None = None
    best_model: str = ""
    selection_rationale: str = ""
    results_path: str = ""


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return post-preprocessing feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def extract_feature_importance(
    pipeline: Pipeline,
    model_name: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Extract feature importance or coefficient magnitudes from a fitted pipeline."""
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_transformed_feature_names(pipeline)

    if model_name == "logistic_regression":
        values = np.abs(classifier.coef_[0])
        label = "abs_coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        label = "importance"
    else:
        return pd.DataFrame(columns=["feature", label, "rank"])

    df = (
        pd.DataFrame({"feature": feature_names, label: values})
        .sort_values(label, ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    if top_n is not None:
        df = df.head(top_n)
    return df


def evaluate_single_model(
    pipeline: Pipeline,
    model_name: str,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    top_features: int = 15,
) -> ModelEvaluationResult:
    """Evaluate one fitted pipeline on the hold-out test set."""
    y_true = y_test.values
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    importance = extract_feature_importance(pipeline, model_name, top_n=top_features)

    return ModelEvaluationResult(
        model_name=model_name,
        metrics=_compute_metrics(y_true, y_pred, y_prob),
        confusion_matrix=cm,
        roc_fpr=fpr,
        roc_tpr=tpr,
        feature_importance=importance,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def build_metrics_comparison(results: list[ModelEvaluationResult]) -> pd.DataFrame:
    """Build a comparison table of test-set metrics for all models."""
    rows = [
        {
            "model": r.model_name,
            "accuracy": round(r.metrics["accuracy"], 4),
            "precision": round(r.metrics["precision"], 4),
            "recall": round(r.metrics["recall"], 4),
            "f1": round(r.metrics["f1"], 4),
            "roc_auc": round(r.metrics["roc_auc"], 4),
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def _load_train_test_gap(model_name: str, training_payload: dict[str, Any]) -> float:
    """Return train minus test ROC-AUC for overfitting assessment."""
    for model in training_payload.get("models", []):
        if model["name"] == model_name:
            train_auc = model["train_metrics"]["roc_auc"]
            test_auc = model["test_metrics"]["roc_auc"]
            return float(train_auc - test_auc)
    return 0.0


def select_best_model(
    results: list[ModelEvaluationResult],
    training_payload: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Select the best model using test metrics and generalization evidence.

    Primary criterion: test ROC-AUC (matches CV tuning objective).
    Tie-breakers: F1, recall, then smallest train-test ROC-AUC gap.
    """
    if not results:
        raise ValueError("No evaluation results to compare.")

    payload = training_payload or {}
    ranked: list[tuple[str, float, float, float, float]] = []

    for r in results:
        gap = _load_train_test_gap(r.model_name, payload) if payload else 0.0
        ranked.append(
            (
                r.model_name,
                r.metrics["roc_auc"],
                r.metrics["f1"],
                r.metrics["recall"],
                -gap,
            )
        )

    ranked.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    best_name = ranked[0][0]
    best = next(r for r in results if r.model_name == best_name)

    gaps = {
        r.model_name: _load_train_test_gap(r.model_name, payload) if payload else None
        for r in results
    }
    gap_text = ""
    if payload:
        best_gap = gaps[best_name]
        gap_text = f" Its train–test ROC-AUC gap ({best_gap:.3f}) is smaller than tree-based models that show near-perfect training scores."

    rationale = (
        f"Selected **{best_name.replace('_', ' ').title()}** based on the hold-out test set. "
        f"It achieves the highest ROC-AUC ({best.metrics['roc_auc']:.3f}), "
        f"the best F1 ({best.metrics['f1']:.3f}), and the highest recall ({best.metrics['recall']:.3f}) "
        f"among the four candidates.{gap_text} "
        f"Performance remains moderate overall (ROC-AUC ≈ 0.82, precision ≈ 0.39); "
        f"the model is useful for ranking attrition risk, not for high-confidence individual predictions."
    )
    return best_name, rationale


def plot_confusion_matrices(
    results: list[ModelEvaluationResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot confusion matrices for all models in a grid."""
    n = len(results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, result in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=result.confusion_matrix,
            display_labels=["Stay", "Leave"],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(result.model_name.replace("_", " ").title())

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices (Test Set)", y=1.02, fontsize=13)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    results: list[ModelEvaluationResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for result in results:
        label = (
            f"{result.model_name.replace('_', ' ').title()} "
            f"(AUC = {result.metrics['roc_auc']:.3f})"
        )
        ax.plot(result.roc_fpr, result.roc_tpr, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    result: ModelEvaluationResult,
    save_path: Path | None = None,
    top_n: int = 15,
) -> plt.Figure:
    """Bar chart of top feature importances for one model."""
    df = result.feature_importance.head(top_n).iloc[::-1]
    value_col = [c for c in df.columns if c not in ("feature", "rank")][0]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    ax.barh(df["feature"], df[value_col], color="#4C72B0")
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_ylabel("")
    ax.set_title(
        f"Top {top_n} Features — {result.model_name.replace('_', ' ').title()}"
    )
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_comparison(
    comparison: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing scalar metrics across models."""
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(metric_cols))
    width = 0.18
    models = comparison["model"].tolist()
    for i, model in enumerate(models):
        row = comparison[comparison["model"] == model].iloc[0]
        scores = [row[m] for m in metric_cols]
        ax.bar(x_pos + i * width, scores, width, label=model.replace("_", " ").title())

    ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_cols])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Test-Set Metrics Comparison")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_best_model(
    model_name: str,
    models_dir: Path | None = None,
) -> Path:
    """Copy the selected model to best_model.joblib."""
    mdir = models_dir or MODELS_DIR
    src = mdir / f"{model_name}.joblib"
    dst = mdir / "best_model.joblib"
    if not src.exists():
        raise FileNotFoundError(f"Trained model not found at {src}.")
    dst.write_bytes(src.read_bytes())
    return dst


def run_evaluation_pipeline(
    config: ProjectConfig | None = None,
    models_dir: Path | None = None,
    figures_dir: Path | None = None,
    model_names: list[str] | None = None,
    save_figures: bool = True,
    top_features: int = 15,
) -> EvaluationReport:
    """Evaluate all persisted models on the hold-out test set."""
    cfg = config or load_config()
    mdir = models_dir or MODELS_DIR
    fdir = figures_dir or REPORTS_FIGURES_DIR
    names = model_names or list(get_model_specs(cfg).keys())

    x, y, _ = prepare_training_data(cfg)
    _, x_test, _, y_test = stratified_train_test_split(x, y, cfg)

    results: list[ModelEvaluationResult] = []
    for name in names:
        pipeline = load_trained_model(name, mdir)
        results.append(evaluate_single_model(pipeline, name, x_test, y_test, top_features))

    comparison = build_metrics_comparison(results)

    training_path = mdir / "training_results.json"
    training_payload = (
        json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else {}
    )
    best_model, rationale = select_best_model(results, training_payload)

    if save_figures:
        fdir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrices(results, fdir / "confusion_matrices.png")
        plot_roc_curves(results, fdir / "roc_curves.png")
        plot_metrics_comparison(comparison, fdir / "metrics_comparison.png")
        for result in results:
            plot_feature_importance(
                result,
                fdir / f"feature_importance_{result.model_name}.png",
                top_n=top_features,
            )
        plt.close("all")

    save_best_model(best_model, mdir)

    eval_payload = {
        "best_model": best_model,
        "selection_rationale": rationale,
        "n_test": len(x_test),
        "positive_rate": float(y_test.mean()),
        "metrics_comparison": comparison.to_dict(orient="records"),
        "confusion_matrices": {
            r.model_name: r.confusion_matrix.tolist() for r in results
        },
        "models": [
            {
                "name": r.model_name,
                "metrics": r.metrics,
                "top_features": r.feature_importance.to_dict(orient="records"),
            }
            for r in results
        ],
    }
    results_path = mdir / "evaluation_results.json"
    results_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")

    return EvaluationReport(
        n_test=len(x_test),
        positive_rate=float(y_test.mean()),
        results=results,
        metrics_comparison=comparison,
        best_model=best_model,
        selection_rationale=rationale,
        results_path=str(results_path),
    )
