---
sidebar_position: 6
---

# Utilities

General utility functions for data manipulation and bias injection.

## Functions

### `drop_data`

```python
def drop_data(
    df: pd.DataFrame,
    percentage: float,
    column1: str,
    value1: Any,
    label_column: str,
    column2: str | None = None,
    value2: Any | None = None,
) -> pd.DataFrame:
```

Drop a percentage of rows from a DataFrame that match specific criteria.

**Parameters:**
- `df`: Input DataFrame.
- `percentage`: Fraction of matching rows to drop (0.0 - 1.0).
- `column1`: First filter column.
- `value1`: Value to match in `column1`.
- `label_column`: Label column name (must be True/1 for row to be dropped).
- `column2`: Optional second filter column.
- `value2`: Optional value to match in `column2`.

---

### `flip_data`

```python
def flip_data(
    df: pd.DataFrame,
    percentage: float,
    column1: str,
    value1: Any,
    label_column: str,
    column2: str | None = None,
    value2: Any | None = None,
) -> pd.DataFrame:
```

Flip the label from True to False for a percentage of rows matching specified criteria.

**Parameters:**
- `df`: Input DataFrame.
- `percentage`: Fraction of matching rows to flip (0.0 - 1.0).
- `column1`: First filter column.
- `value1`: Value to match in `column1`.
### `balance_data`

```python
def balance_data(
    df: pd.DataFrame,
    column1: str,
    label_column: str,
) -> tuple[pd.DataFrame, int]:
```

Balances both group sizes and selection rates across groups defined by `column1`. Ensures every group has the same number of positive and negative samples by undersampling.

**Returns:** A tuple containing the balanced DataFrame and the total number of samples removed.

---

### `generate_modification_dict`

```python
def generate_modification_dict(
    client_ids: int | list[int | str],
    attribute: str,
    value: Any,
    drop_rate_range: tuple[float, float] = (0.0, 0.0),
    flip_rate_range: tuple[float, float] = (0.0, 0.0),
    secondary_attribute: str | None = None,
    secondary_value: Any | None = None,
) -> dict[Any, dict[str, Any]]:
```

Generates a modification dictionary for simulating data heterogeneity across clients. Linearly interpolates the modification rates across the provided clients.

---

### `generate_bias_by_groups`

```python
def generate_bias_by_groups(
    num_total_clients: int, 
    group_configs: list[dict[str, Any]], 
    client_names: list[str | int] | None = None
) -> dict[Any, dict[str, Any]]:
```

Generates a `modification_dict` by partitioning clients into groups and sampling bias rates from truncated normal distributions.

---

### `generate_multiobjective_bias`

```python
def generate_multiobjective_bias(
    num_total_clients: int,
    group_configs: list[dict[str, Any]],
    client_names: list[str | int] | None = None,
) -> dict[Any, dict[str, Any]]:
```

Generates a modification dictionary for multi-objective fairness scenarios. Allows defining multiple attribute modifications per group (simultaneous mitigation and bias injection).
