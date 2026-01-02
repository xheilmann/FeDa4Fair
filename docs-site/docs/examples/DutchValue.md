# Value-Based Fairness Example (Value Skew)

This example demonstrates how different client groups can favor different values of the same sensitive attribute. For instance, Group A might be biased against Males while Group B is biased against Females.

## Scenario

We want to create a federation where:
1.  **Group A**: Biased AGAINST value 0 (favors 1).
2.  **Group B**: Biased AGAINST value 1 (favors 0).

## Group Configuration

We use `generate_multiobjective_bias` with aggressive bias injection to ensure the model learns the skewed patterns.

```python
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias

num_clients = 100
group_split = 50
client_names = [str(i) for i in range(num_clients)]

# Configure groups to show opposite bias directions for 'sex_binary'
group_configs = [
    {
        "group_id": "Group A (Favors 1)",
        "num_clients": group_split,
        "configs": [
            {
                "attribute": "sex_binary",
                "value": 0,  # Bias against 0 (favors 1)
                "drop_mean": 0.9, "drop_std": 0.05,
                "flip_mean": 0.3, "flip_std": 0.05
            }
        ]
    },
    {
        "group_id": "Group B (Favors 0)",
        "num_clients": num_clients - group_split,
        "configs": [
            {
                "attribute": "sex_binary",
                "value": 1,  # Bias against 1 (favors 0)
                "drop_mean": 0.3, "drop_std": 0.05,
                "flip_mean": 0.1, "flip_std": 0.05
            }
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
    sensitive_attributes=["sex_binary"],
    modification_dict=modifications,
    fl_setting="cross-silo",
    perc_train_val_test=[0.8, 0.2],
    client_names=client_names
)

fds.prepare()
```

## Visualization with Signed-Bias Coloring

The library can automatically color-code bars based on which group is favored.

```python
from FeDa4Fair.visualization import plot_multi_attribute_fairness
from sklearn.linear_model import LogisticRegression

# Red for favoring group 0, Blue for favoring group 1
val_colors = {0: "red", 1: "blue"}

fig, ax, results = plot_multi_attribute_fairness(
    partitioner=fds.partitioners["train_train"],
    partitioner_test=fds.partitioners["train_test"],
    label_name="occupation_binary",
    sens_atts=["sex_binary"],
    fairness_metric="DP",
    model=LogisticRegression(max_iter=1000, solver="liblinear"),
    size_unit="value",
    value_colors=val_colors,
    fds=fds,
    split="train_train",
    test_split="train_test",
    title="Model Prediction Fairness (Favored Group Color Coding)"
)
```