---
sidebar_position: 3
---

# Using Custom Datasets

FeDa4Fair is designed to be flexible and supports loading not just the built-in ACS datasets, but any dataset available on the Hugging Face Hub or local datasets (via Hugging Face's loading scripts).

This guide explains how to use an external dataset, specifically focusing on the `criteo/FairJob` dataset as an example.

## Loading a Hugging Face Dataset

To use a custom dataset, you instantiate the `FairFederatedDataset` class with the name of the dataset on the Hugging Face Hub. You must also specify the column names that correspond to the target label and the sensitive attributes you wish to analyze.

### Key Parameters

- **`dataset`**: The path or name of the dataset on Hugging Face (e.g., `"criteo/FairJob"`).
- **`label_name`**: The name of the column containing the target variable (e.g., `"senior"`).
- **`sensitive_attributes`**: A list of column names representing the sensitive attributes (e.g., `["protected_attribute"]`).
- **`partitioners`**: A dictionary defining how to split the data. You can specify the number of partitions (iid) for a given split (e.g., `{"train": 10}`).

### Example: Criteo FairJob Dataset

In this example, we will load the `criteo/FairJob` dataset. We identify `protected_attribute` as the sensitive attribute and `senior` as the target label.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

# Initialize the FairFederatedDataset with the custom dataset
fds = FairFederatedDataset(
    dataset="criteo/FairJob",
    partitioners={"train": 10},  # Split the 'train' split into 10 partitions
    label_name="senior",         # The target label
    sensitive_attributes=["protected_attribute"], # The sensitive attribute to consider
    fairness_metric="DP",        # Metric to evaluate (e.g., Demographic Parity)
)

# Prepare the dataset (downloads and partitions data)
fds.prepare()

# Access a specific partition
partition_0 = fds.load_partition(0, "train")
print(f"Partition 0 size: {len(partition_0)}")

# Example: Check the distribution of the sensitive attribute in the first partition
df_0 = partition_0.to_pandas()
print(df_0["protected_attribute"].value_counts(normalize=True))
```

## Using Local Datasets

You can also use local datasets by leveraging Hugging Face's support for local files (CSV, JSON, Parquet, ImageFolder).

```python
fds = FairFederatedDataset(
    dataset="csv",
    data_files={"train": "path/to/my_data.csv"},
    partitioners={"train": 5},
    label_name="target_col",
    sensitive_attributes=["gender", "race"]
)
fds.prepare()
```
