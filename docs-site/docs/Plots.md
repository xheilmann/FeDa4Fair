---
sidebar_position: 5
---

# Plots

This module contains specialized plotting functions for visualizing fairness metrics and label distributions across federated learning partitions.

## Functions

### `plot_comparison_label_distribution`

```python
def plot_comparison_label_distribution(
    partitioner_list: list[Partitioner],
    label_name: str | list[str],
    plot_type: Literal["bar", "heatmap"] = "bar",
    # ...
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
```

Compare the label distribution across multiple partitioners. Useful for visualizing non-IID data distributions.

---

### `plot_comparison_fairness_distribution`

```python
def plot_comparison_fairness_distribution(
    partitioner_dict: dict[str, Partitioner],
    label_name: str = "ECP",
    sens_att: str = "SEX",
    fairness_metric: Literal["DP", "EO"] = "DP",
    # ...
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
```

Compare fairness metric distributions across multiple partitioners. Generates heatmaps or bar charts showing fairness metrics per partition.

---

### `plot_fairness_distributions`

```python
def plot_fairness_distributions(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    label_name: str,
    sens_att: str,
    fairness_metric: Literal["DP", "EO"] = "DP",
    # ...
) -> tuple[Figure, Axes, pd.DataFrame]:
```

Plot fairness metric distributions for a single partitioner (and its test set) across its partitions.
