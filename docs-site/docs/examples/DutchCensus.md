# Dutch Census Example

This example shows how to use the `lucacorbucci/Dutch_Census` dataset from Hugging Face. It demonstrates a simple federated setup without initial bias injection.

## Basic Setup

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

# Initialize with 10 IID partitions
fds = FairFederatedDataset(
    dataset="lucacorbucci/Dutch_Census",
    partitioners={"train": 10},
    label_name="occupation_binary",
    sensitive_attributes=["sex_binary"],
    fairness_metric="DP"
)

fds.prepare()
```

## Accessing Partitions

You can load individual client partitions for local training.

```python
# Load partition for client 0
client_0 = fds.load_partition(0, split="train")
print(f"Client 0 size: {len(client_0)}")
```
