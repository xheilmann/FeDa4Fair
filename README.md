# FeDa4Fair

Federated Learning (FL) enables collaborative model training across multiple clients while preserving the privacy of their local data. However, fairness remains a critical concern, as inherent biases within individual clients' datasets may influence the entire federated system. In particular, the presence of various clients, each with different data distributions, brings with it the risk that the trained federated model may result fairer for some groups of clients than for others. Although several fairness-enhancing strategies have been proposed in the literature, most focus on mitigating bias for a single sensitive attribute, typically binary, without addressing the diverse and sometimes conflicting fairness needs of different clients. This limited perspective may result in fairness interventions that fail to produce meaningful improvements for all clients. We aim to improve the study of fairness mitigation and evaluation in FL by allowing reproducible and consistent benchmarking of fairness-aware FL methods, globally and at the client level. Therefore, we introduce FeDa4Fair, a library to generate tabular datasets specifically designed to evaluate fair FL methods, encompassing diverse heterogeneous client scenarios with respect to bias in sensitive attributes. Additionally, we release 4 benchmarking datasets. We also provide ready-to-use functions for evaluating fairness outcomes for these datasets.

## Create the environment

First of all we need to install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then we can create the environment:

```bash
uv sync
uv venv
```

We suggest you create the helper folder `data_stats`:

```bash
mkdir src/FeDa4Fair/data_stats
```

How to run code that creates an example dataset:

```bash
uv run python src/FeDa4Fair/main.py
```

## Tutorial and Example 

A detailed example/tutorial on how to use the library can be found in [example.ipynb](src/FeDa4Fair/example.ipynb).


## Run Formatting 

```bash
uv run ruff format
```

## Benchmarking Datasets

We provide 4 benchmarking datasets and their corresponding datasheets:
1. **Attribute-silo** dataset: a dataset which can be used in cross-silo settings where the attribute bias varies over clients ([attribute-silo](src/FeDa4Fair/data/cross_silo_attribute_final))
2. **Attribute-device** dataset: a dataset which can be used in cross-device settings where the attribute bias varies over clients ([attribute-device](src/FeDa4Fair/data/cross_device_attribute_final))
3. **Value-silo** dataset: a dataset which can be used in cross-silo settings where the value bias varies over clients for the RACE attribute ([value-silo](src/FeDa4Fair/data/cross_silo_value_final))
4. **Value-device** dataset: a  dataset which can be used in cross-device settings where the value bias varies over clients for the RACE attribute ([value-device](src/FeDa4Fair/data/cross_device_value_final))

## Using Generic Datasets (Hugging Face & Local)

FeDa4Fair supports loading arbitrary datasets from Hugging Face Hub or local files.

### 1. Hugging Face Datasets

You can load any dataset available on Hugging Face Hub by specifying its name.

```python
from FairFederatedDataset import FairFederatedDataset

fds = FairFederatedDataset(
    dataset="lucacorbucci/Dutch_Census",
    partitioners={"train": 10},  # Split train set into 10 partitions
    label_name="occupation_binary",
    sensitive_attributes=["sex_binary"],
    fairness_metric="DP"
)
fds._prepare_dataset()
```

See `examples/dutch_census_example.py` for a complete example.

### 2. Local Image Datasets

To use a local dataset (e.g., images), organize your data in a folder structure supported by Hugging Face `imagefolder` or `folder` builders, or simply point to the directory if it contains metadata.

```python
fds = FairFederatedDataset(
    dataset="imagefolder",
    data_dir="/path/to/local/data",
    partitioners={"train": 2},
    label_name="label",
    sensitive_attributes=["sensitive_attr"],
)
```

See `examples/celeba_example.py` for an example using a mock local dataset.