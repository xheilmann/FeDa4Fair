# Multi-Objective Fairness Example (Attribute Skew)

This example demonstrates how to configure groups that simultaneously mitigate bias for one attribute and inject bias for another. This is useful for studying conflicting fairness goals across a federation.

## Scenario

We want to create two groups of clients using the Dutch Census dataset:
1.  **Group A**: Mitigates bias toward `sex_binary` but has injected bias toward `Marital_status`.
2.  **Group B**: Mitigates bias toward `Marital_status` but has injected bias toward `sex_binary`.

## Group Configuration

We use the `generate_multiobjective_bias` utility to define these complex profiles sampled from Normal distributions.

```python
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias

num_clients = 100
group_split_idx = 50 
client_names = [str(i) for i in range(num_clients)]

# Define multiple attribute modifications per group
group_configs = [
    {
        "group_id": "Group A",
        "num_clients": group_split_idx,
        "configs": [
            # Mitigate sex_binary
            {"attribute": "sex_binary", "mitigate": True},
            # Bias Marital_status (Drop value 0)
            {
                "attribute": "Marital_status",
                "value": 0,
                "drop_mean": 0.3, "drop_std": 0.1,
                "flip_mean": 0.2, "flip_std": 0.05
            }
        ]
    },
    {
        "group_id": "Group B",
        "num_clients": num_clients - group_split_idx,
        "configs": [
            # Bias sex_binary (Drop value 1)
            {
                "attribute": "sex_binary",
                "value": 1,
                "drop_mean": 0.4, "drop_std": 0.1,
                "flip_mean": 0.3, "flip_std": 0.05
            },
            # Mitigate Marital_status
            {"attribute": "Marital_status", "mitigate": True}
        ]
    }
]

modifications = generate_multiobjective_bias(num_clients, group_configs, client_names)
```

## Dataset Initialization

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

fds = FairFederatedDataset(
    dataset="lucacorbucci/Dutch_census_binary_marital_status",
    split="all",
    partitioners={"train": num_clients},
    label_name="occupation_binary",
    sensitive_attributes=["sex_binary", "Marital_status"],
    modification_dict=modifications,
    fl_setting="cross-silo",
    perc_train_val_test=[0.8, 0.2],
    client_names=client_names
)

print("Preparing dataset...")
fds.prepare()
```

## Visualization

You can use the library's built-in plotting functions to verify the injected bias profiles.

```python
from FeDa4Fair.visualization import plot_multi_attribute_fairness
from sklearn.linear_model import LogisticRegression

# Evaluate Model Demographic Parity for both attributes side-by-side
fig, ax, results = plot_multi_attribute_fairness(
    partitioner=fds.partitioners["train_train"],
    partitioner_test=fds.partitioners["train_test"],
    label_name="occupation_binary",
    sens_atts=["sex_binary", "Marital_status"],
    fairness_metric="DP",
    model=LogisticRegression(max_iter=1000, solver="liblinear"),
    max_num_partitions=20,
    title="Model DP by Attribute (Subset of Clients)",
    fds=fds,
    split="train_train",
    test_split="train_test"
)
```