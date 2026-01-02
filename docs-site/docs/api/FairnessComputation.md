---
sidebar_position: 3
---

# Fairness Computation

This module contains the core logic for computing fairness metrics such as Demographic Parity (DP) and Equalized Odds (EO).

## Functions

### `compute_fairness`

```python
def compute_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    model: Any,
    sens_att: str,
    max_num_partitions: int | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    label_name: str = "label",
    sens_cols: list[str] | None = None,
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
) -> pd.DataFrame:
```

Computes fairness metrics across dataset partitions.

**Parameters:**
- `partitioner` (*Partitioner*): Training/reference data partitioner.
- `partitioner_test` (*Partitioner*): Test data partitioner.
- `model` (*Any*): Model to evaluate (optional).
- `sens_att` (*str*): Sensitive attribute column name.
- `max_num_partitions` (*Optional[int]*): Limit on partitions to evaluate.
- `fairness_metric` (*str*, default="DP"): "DP" or "EO".
- `label_name` (*str*, default="label"): Target label column name.
- `sens_cols` (*Optional[list[str]]*): Sensitive attributes to drop before training.
- `size_unit` (*str*, default="attribute"): Detail level of result.

**Returns:**
- *pd.DataFrame*: Fairness metrics for each partition.

---

### `_compute_fairness` (Internal)

```python
def _compute_fairness(
    y_true: Any,
    y_pred: Any,
    sf_data: pd.DataFrame,
    fairness_metric: Literal["DP", "EO"],
    sens_att: str,
    size_unit: Literal["value", "attribute", "attribute-value"],
) -> pd.Series:
```

Compute a fairness metric (Demographic Parity or Equalized Odds) for given sensitive attribute(s).

**Parameters:**
- `y_true`: Ground truth labels.
- `y_pred`: Model predictions.
- `sf_data`: DataFrame containing sensitive feature(s).
- `fairness_metric`: "DP" or "EO".
- `sens_att`: Sensitive attribute name.
- `size_unit`: Level of detail.

---

### `compute_multi_fairness`

```python
def compute_multi_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    model: Any,
    sens_atts: list[str],
    max_num_partitions: int | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    label_name: str = "label",
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
) -> pd.DataFrame:
```

Computes fairness metrics for multiple sensitive attributes independently in a single pass. 

**Key Benefits:**
- **Performance:** Trains the model only once per partition, then evaluates it against all provided sensitive attributes.
- **Side-by-Side:** Returns a DataFrame where each row is a partition and columns represent the fairness metric for each attribute.
