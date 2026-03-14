"""Fairness metrics and evaluation utilities."""

from FeDa4Fair.metrics.evaluation import evaluate_fairness, evaluate_models_on_datasets
from FeDa4Fair.metrics.fairness import compute_fairness, compute_multi_fairness

__all__ = [
    "compute_fairness",
    "compute_multi_fairness",
    "evaluate_fairness",
    "evaluate_models_on_datasets",
]
