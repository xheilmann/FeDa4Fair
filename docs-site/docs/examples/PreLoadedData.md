# Pre-Loaded Data Example

This example demonstrates how to use `FairFederatedDataset` with data that has been pre-processed independently as a Pandas DataFrame.

## Usage

You can pass a single DataFrame directly to the `preloaded_data` parameter.

```python
import pandas as pd
from FeDa4Fair.dataset import FairFederatedDataset

# Load and preprocess your data
df = pd.read_csv("my_custom_data.csv")
df["processed_col"] = df["raw_col"] * 2

# Initialize FairFederatedDataset with pre-loaded data
fds = FairFederatedDataset(
    dataset="custom_name",
    preloaded_data=df, 
    partitioners={"train": 5},
    label_name="label",
    sensitive_attributes=["sensitive_col"]
)

fds.prepare()
```
