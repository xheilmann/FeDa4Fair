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

---

### `plot_multi_attribute_fairness`

```python
def plot_multi_attribute_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    label_name: str,
    sens_atts: list[str],
    fairness_metric: Literal["DP", "EO"] = "DP",
    model: Any | None = None,
    size_unit: Literal["value", "attribute"] = "attribute",
    value_colors: dict[Any, str] | None = None,
    # ...
) -> tuple[Figure, Axes, pd.DataFrame]:
```

Plot fairness metrics for multiple sensitive attributes side-by-side for each partition. 

**Key Features:**
- **Grouped Bar Charts:** Compare different attributes (e.g., SEX and MAR) per client.
- **Signed-Bias Coloring:** When `size_unit='value'`, bars are color-coded based on which group is favored (using the `value_colors` mapping).
- **Red/Blue Coloring:** Automatically highlights conflicting bias directions across clients.
