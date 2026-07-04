"""Model training, evaluation, and persistence."""

from attrisense.models.evaluation import (
    EvaluationReport,
    ModelEvaluationResult,
    build_metrics_comparison,
    evaluate_single_model,
    extract_feature_importance,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_roc_curves,
    run_evaluation_pipeline,
    select_best_model,
)
from attrisense.models.pipelines import build_model_pipeline, build_preprocessor, split_feature_groups
from attrisense.models.training import (
    ModelTrainingResult,
    TrainingReport,
    get_model_specs,
    load_selected_features,
    load_trained_model,
    prepare_training_data,
    run_training_pipeline,
    stratified_train_test_split,
    train_single_model,
)

__all__ = [
    "EvaluationReport",
    "ModelEvaluationResult",
    "ModelTrainingResult",
    "TrainingReport",
    "build_metrics_comparison",
    "build_model_pipeline",
    "build_preprocessor",
    "evaluate_single_model",
    "extract_feature_importance",
    "get_model_specs",
    "load_selected_features",
    "load_trained_model",
    "plot_confusion_matrices",
    "plot_feature_importance",
    "plot_metrics_comparison",
    "plot_roc_curves",
    "prepare_training_data",
    "run_evaluation_pipeline",
    "run_training_pipeline",
    "select_best_model",
    "split_feature_groups",
    "stratified_train_test_split",
    "train_single_model",
]
