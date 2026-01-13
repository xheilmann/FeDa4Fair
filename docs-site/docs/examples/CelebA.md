# CelebA Image Example

This example demonstrates how to use the `flwrlabs/celeba` image dataset from Hugging Face with FeDa4Fair. It covers merging splits and applying group-based bias injection for image metadata.

## Configuration

We simulate a scenario with 150 clients partitioned into two groups with distinct bias profiles regarding the "Male" attribute.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset
from FeDa4Fair.utils.data_utils import generate_bias_by_groups

num_total_clients = 150

# Define bias groups
group_configs = [
    {
        "group_id": "Group A (Unfair to Female)",
        "num_clients": 75,
        "sensitive_attr": "Male",
        "sensitive_value": 0, # Female
        "drop_mean": 0.2, "drop_std": 0.05,
        "flip_mean": 0.1, "flip_std": 0.02,
    },
    {
        "group_id": "Group B (Unfair to Male)",
        "num_clients": 75,
        "sensitive_attr": "Male",
        "sensitive_value": 1, # Male
        "drop_mean": 0.3, "drop_std": 0.05,
        "flip_mean": 0.2, "flip_std": 0.02,
    }
]

mod_dict = generate_bias_by_groups(num_total_clients, group_configs)

# Initialize dataset
# By specifying split="all", we merge train, validation, and test splits from HF
fds = FairFederatedDataset(
    dataset="flwrlabs/celeba",
    split="all", 
    partitioners={"train": num_total_clients},
    modification_dict=mod_dict,
    label_name="Smiling",
    sensitive_attributes=["Male"]
)

print("Preparing dataset...")
fds.prepare()
```

## Integration with PyTorch

You can iterate through the partitions and convert them to your desired format. FeDa4Fair provides a `TabularDataset` utility for PyTorch, but for images, you can use the raw Hugging Face `Dataset` objects returned by `load_partition`.

```python
# Access partition for client 0
client_0_data = fds.load_partition(0, split="train")
print(f"Number of samples for client 0: {len(client_0_data)}")
```