# FeDa4Fair Usage Examples

This guide provides examples of how to use the FeDa4Fair library with different types of datasets and how to launch the interactive dashboard.

## 1. Setup

Ensure you have `uv` installed and the environment synchronized:

```bash
uv sync
```

## 2. Using Hugging Face Datasets

FeDa4Fair can now load any dataset from the Hugging Face Hub.

### Example: Dutch Census Dataset
The following script demonstrates how to load the Dutch Census dataset, partition it into multiple clients, and evaluate data bias (Demographic Parity).

**Run the example:**
```bash
uv run python examples/dutch_census_example.py
```

**Code Snippet:**
```python
from FairFederatedDataset import FairFederatedDataset

fds = FairFederatedDataset(
    dataset="lucacorbucci/Dutch_Census",
    partitioners={"train": 5}, # 5 clients
    label_name="occupation_binary",
    sensitive_attributes=["sex_binary"],
    fairness_metric="DP"
)
fds._prepare_dataset()
fds.evaluate()
```

## 3. Using Local Image Datasets

You can use local datasets by utilizing Hugging Face's `imagefolder` builder.

### Example: CelebA (Mock)
We provide a mock generator to test this feature without downloading the full CelebA dataset.

1. **Generate Mock Data:**
   ```bash
   uv run python examples/generate_mock_data.py
   ```
2. **Run Image Example:**
   ```bash
   uv run python examples/celeba_example.py
   ```

This example shows how to use the `ImageDataset` wrapper and `SimpleCNN` provided in `example_utils.py`.

## 4. Interactive Dashboard

We have included a Streamlit dashboard to interactively configure datasets, partitioning strategies, and bias injection.

**Launch the dashboard:**
```bash
uv run streamlit run dashboard/app.py
```

**Features:**
- Select between ACS datasets and Hugging Face datasets.
- Adjust the number of clients and partitioning strategy (IID vs Dirichlet).
- Inject synthetic bias by dropping or flipping data for specific groups.
- Visualize the resulting fairness metrics across clients.

## 5. Running Tests

To ensure everything is working correctly, run the unit tests:

```bash
uv run python tests/test_fair_federated_dataset.py
```

## 6. Project Structure Overview

- `src/FeDa4Fair/FairFederatedDataset.py`: Core class for dataset management.
- `src/FeDa4Fair/fairness_computation.py`: Logic for computing DP and EO metrics.
- `src/FeDa4Fair/example_utils.py`: PyTorch wrappers for tabular and image data.
- `src/FeDa4Fair/utils.py`: Data modification helpers (drop/flip).
- `examples/`: Example scripts for different datasets.
- `dashboard/`: Streamlit application code.
