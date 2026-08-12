"""Core DJGP projection implementation."""

from djgp.projections.structured_metric_lmjgp import (
    AnalyticStructuredMetricLMJGPConfig,
    build_fixed_local_inducing_from_w0,
    predict_analytic_structured_metric_lmjgp,
    train_analytic_structured_metric_lmjgp,
)

__all__ = [
    "AnalyticStructuredMetricLMJGPConfig",
    "build_fixed_local_inducing_from_w0",
    "predict_analytic_structured_metric_lmjgp",
    "train_analytic_structured_metric_lmjgp",
]
