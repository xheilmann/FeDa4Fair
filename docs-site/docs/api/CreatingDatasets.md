---
sidebar_position: 2
---

# Creating Datasets

This module provides utilities for generating and preprocessing datasets for different federated learning settings (cross-silo and cross-device), including bias injection and evaluation.

## Functions

### `create_cross_silo_data`

```python
def create_cross_silo_data(fairness_level: str, path: str) -> None:
```

Generate, evaluate, and save cross-silo datasets with varying bias levels.

**Parameters:**
- `fairness_level` (*str*): The fairness level to evaluate (e.g., "attribute", "value").
- `path` (*str*): The base path for saving data and stats.

---

### `preprocess_data_cross_silo`

```python
def preprocess_data_cross_silo(data1: pd.DataFrame, datasets: list, fairness_level: str, state: str) -> list:
```

Preprocess data for cross-silo federated learning by splitting into train/test and extracting sensitive features.

**Parameters:**
- `data1` (*pd.DataFrame*): Input DataFrame for a specific state/silo.
- `datasets` (*list*): Accumulator list of processed datasets.
- `fairness_level` (*str*): Fairness level.
- `state` (*str*): State identifier.

---

### `create_cross_device_data`

```python
def create_cross_device_data(fairness_level: str, split_number: int, path: str) -> None:
```

Generate, evaluate, and save cross-device datasets based on previously generated cross-silo data.

**Parameters:**
- `fairness_level` (*str*): Fairness level to evaluate.
- `split_number` (*int*): Number of splits per state (simulating devices).
- `path` (*str*): Base path.

---

### `preprocess_datasets`

```python
def preprocess_datasets(file: str, data1: pd.DataFrame, path: str, split_number: int = 6, fairness_level: str = "attribute") -> list:
```

Split a dataset into multiple parts and preprocess each for cross-device evaluation.

**Parameters:**
- `file` (*str*): Filename or identifier.
- `data1` (*pd.DataFrame*): Input DataFrame.
- `path` (*str*): Base path.
- `split_number` (*int*, default=6): Number of splits.
- `fairness_level` (*str*, default="attribute"): Fairness level.

---

### `split_df`

```python
def split_df(df: pd.DataFrame, split_number: int) -> list[pd.DataFrame]:
```

Split a DataFrame into a specified number of approximately equal parts.

---

### `generate_bias_by_groups`

```python
def generate_bias_by_groups(
    num_total_clients: int, 
    group_configs: list[dict[str, Any]], 
    client_names: list[str | int] | None = None
) -> dict[Any, dict[str, Any]]:
```

Generates a `modification_dict` by partitioning clients into groups and sampling bias rates from **Truncated Normal Distributions** bounded between `[0, 1]`.

**Parameters:**
- `num_total_clients` (*int*): The total number of clients in the federation.
- `group_configs` (*list[dict]*): List of group configurations. Each config should contain:
    - `group_id`: Identifier for the group.
    - `num_clients`: Number of clients in this group.
    - `sensitive_attr`: The attribute to target.
    - `sensitive_value`: The value to disadvantage.
    - `drop_mean` / `drop_std`: Distribution parameters for dropping data.
    - `flip_mean` / `flip_std`: Distribution parameters for flipping labels.
- `client_names` (*Optional[list]*): Custom names for clients.

**Returns:**
- A dictionary mapping client IDs to their specific sampled data modifications.
