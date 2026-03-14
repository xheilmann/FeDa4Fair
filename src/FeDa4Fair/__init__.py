"""FeDa4Fair — Federated Data for Fairness."""

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset
from FeDa4Fair.dataset.generation import (
    create_cross_device_data,
    create_cross_silo_data,
    generate_bias_modifications,
)
from FeDa4Fair.dataset.partitioning import RepresentativeDiversityPartitioner
from FeDa4Fair.metrics.evaluation import evaluate_fairness, evaluate_models_on_datasets
from FeDa4Fair.metrics.fairness import compute_fairness, compute_multi_fairness
from FeDa4Fair.utils.data_utils import (
    balance_data,
    cap_samples,
    drop_data,
    flip_data,
    generate_bias_by_groups,
    generate_modification_dict,
    generate_multiobjective_bias,
)

__all__ = [
    "FairFederatedDataset",
    "RepresentativeDiversityPartitioner",
    "balance_data",
    "cap_samples",
    "compute_fairness",
    "compute_multi_fairness",
    "create_cross_device_data",
    "create_cross_silo_data",
    "drop_data",
    "evaluate_fairness",
    "evaluate_models_on_datasets",
    "flip_data",
    "generate_bias_by_groups",
    "generate_bias_modifications",
    "generate_modification_dict",
    "generate_multiobjective_bias",
]
