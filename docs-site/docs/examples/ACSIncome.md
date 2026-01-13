# ACSIncome Example

This example demonstrates how to use the FeDa4Fair library with the ACSIncome dataset from Folktables. It covers generating a federated dataset with multiple states and performing fairness evaluations.

## Setup

To generate a dataset and its partitions, you specify the dataset name ("ACSIncome" or "ACSEmployment") and the states you want to load.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

# Initialize FairFederatedDataset with multiple states
fds = FairFederatedDataset(
    dataset="ACSIncome",
    states=["CT", "AK"],
    partitioners={"CT": 2, "AK": 2},
    sensitive_attributes=["SEX", "RAC1P"],
    fairness_metric="DP"
)

# Prepare the dataset (triggers download and partitioning)
fds.prepare()
```

## Evaluation

Once prepared, you can evaluate the fairness metrics across all partitions.

```python
# Run fairness evaluation
fds.evaluate()
```
