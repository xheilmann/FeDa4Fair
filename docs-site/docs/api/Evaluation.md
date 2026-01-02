---
sidebar_position: 4
---

# Evaluation

This module provides high-level functions to evaluate fairness and visualize distributions across partitions and models.

## Functions

### `evaluate_fairness`

```python
def evaluate_fairness(
    partitioner_dict: dict[str, Partitioner],
    max_num_partitions: int | None = 10,
    sens_columns: str | list[str] | None = None,
    intersectional_fairness: list[str] | None = None,
    size_unit: Literal["percent", "absolute"] = "absolute",
    fairness_metric: Literal["DP", "EO"] = "DP",
    fairness_level: Literal["attribute", "value", "attribute-value"] = "attribute",
    # ... plotting arguments ...
) -> None:
```

Save, evaluate, and visualize fairness metrics and data counts across partitions.

**Key Parameters:**
- `partitioner_dict`: Dictionary of partitioners.
- `sens_columns`: Sensitive attribute(s) to evaluate.
- `fairness_metric`: "DP" or "EO".
- `fairness_level`: Level of fairness evaluation.
- `model`: Model object (optional).

---

### `evaluate_models_on_datasets`

```python
def evaluate_models_on_datasets(
    datasets: list[tuple], 
    n_jobs: int = -1, 
    fairness_metric: str = "DP", 
    fairness_level: str = "attribute"
) -> tuple[pd.DataFrame, Any]:
```

Evaluates multiple models (LogisticRegression, XGBoost) on multiple datasets in parallel.

**Parameters:**
- `datasets`: List of dataset tuples (name, X_train, y_train, X_test, y_test, sensitive_features).
- `n_jobs`: Number of parallel jobs.
- `fairness_metric`: "DP" or "EO".
- `fairness_level`: "attribute", "value", etc.

**Returns:**
- *tuple*: Results DataFrame and the plot object.

---

### `evaluate_model`

```python
def evaluate_model(
    model_name: str,
    model: Any,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    fairness_metric: Literal["DP", "EO"],
    sf_data: dict[str, np.ndarray],
    fairness_level: str,
) -> dict:
```

Trains and evaluates a single classification model, returning accuracy and fairness metrics.

---

### `local_client_fairness_plot`

```python
def local_client_fairness_plot(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    client_column: str = "Partition ID",
    fairness_column: str = "RAC1P_DP",
    # ...
) -> plt.Figure:
```

Plot a scatter comparison of fairness values from two dataframes (e.g., before and after an intervention).
