---
sidebar_position: 1
---

# FairFederatedDataset

`FairFederatedDataset` is the core class of the FeDa4Fair library. It extends `flwr_datasets.FederatedDataset` to provide enhanced capabilities for fairness evaluation, bias injection, and dataset management in federated learning simulations.

## Class Definition

### `FairFederatedDataset`

```python
class FairFederatedDataset(FederatedDataset):
    ...
```

Representation of a dataset designed for federated learning, fairness evaluation, and analytics.

Supports downloading, loading, preprocessing, modifying, evaluating, mapping, and partitioning the dataset across multiple clients (e.g., edge devices or simulated silos).

#### Parameters

- **`dataset`** (*str*, default="ACSIncome"): The name of the dataset to load. Supports "ACSIncome", "ACSEmployment" (via folktables), or any Hugging Face dataset name.
- **`subset`** (*Optional[str]*, default=None): Optional dataset subset to load.
- **`split`** (*Optional[str]*): The split to load (e.g., "train", "test"). Set to **"all"** to load and concatenate all available splits into a single dataset.
- **`preprocessor`** (*Optional[Union[Preprocessor, dict]]*, default=None): Transformations to apply on the dataset.
- **`partitioners`** (*dict[str, Union[Partitioner, int]]*): Dictionary mapping splits to partitioning strategies.
- **`shuffle`** (*bool*, default=True): Whether to shuffle the dataset.
- **`seed`** (*Optional[int]*, default=42): Random seed.
- **`states`** (*Optional[list[str]]*, default=None): List of states for ACS datasets.
- **`year`** (*Optional[str]*, default="2018"): ACS year.
- **`horizon`** (*Optional[str]*, default="1-Year"): ACS horizon.
- **`sensitive_attributes`** (*Optional[list[str]]*, default=None): Attributes for intersectional fairness.
- **`fairness_level`** (*Literal["attribute", "value", "attribute-value"]*, default="attribute"): Fairness evaluation level.
- **`fairness_metric`** (*Literal["DP", "EO"]*, default="DP"): Fairness metric (Demographic Parity / Equalized Odds).
- **`fl_setting`** (*Literal["cross-silo", "cross-device", None]*, default=None): FL strategy.
- **`perc_train_val_test`** (*Optional[list[float]]*, default=None): Split proportions.
- **`path`** (*Optional[PathLike]*, default=None): Path to save the dataset.
- **`modification_dict`** (*Optional[dict]*, default=None): Configuration for bias injection.
- **`mapping`** (*Optional[dict]*, default=None): Remapping dictionary for features/labels.
- **`label_name`** (*Optional[str]*, default=None): Target label column name.
- **`preloaded_data`** (*Optional[dict]*, default=None): Dictionary of pre-loaded DataFrames.

#### Methods

- **`prepare()`**: Explicitly triggers dataset preparation.
- **`save_dataset(dataset_path)`**: Saves the dataset partitions to CSV files.
- **`evaluate(file=None)`**: Evaluates fairness on all partitions.
- **`load_acs_raw_data(dataset_name, states, year, horizon)`** (*staticmethod*): Loads raw ACS data in parallel.

## Usage Example

```python
from FairFederatedDataset import FairFederatedDataset

ffds = FairFederatedDataset(
    dataset="ACSIncome",
    states=["CA", "NY"],
    partitioners={"CA": 5, "NY": 5},
    fairness_metric="DP"
)
ffds.prepare()
```
