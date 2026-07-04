"""AttriSense — employee attrition prediction and HR analytics.

Public API surface for the core package. Typical entry points:

- ``attrisense.config.load_config`` — project configuration
- ``attrisense.data.run_preprocessing_pipeline`` — data preparation
- ``attrisense.data.run_feature_engineering_pipeline`` — feature creation
- ``attrisense.models.run_training_pipeline`` — model tuning and persistence
- ``attrisense.models.run_evaluation_pipeline`` — metrics and model selection
- ``attrisense.inference.predict_attrition`` — scoring at inference time

See ``docs/ARCHITECTURE.md`` for system design and ``docs/WORKFLOW.md`` for
end-to-end usage.
"""

__version__ = "0.1.0"
