# FeDa4Fair

<p align="center">
  <img src="img/logo_no_background.png" width="200" alt="FeDa4Fair Logo">
</p>

**FeDa4Fair** is a comprehensive library designed to facilitate the study of fairness in Federated Learning (FL). It allows researchers and practitioners to generate, manipulate, and benchmark tabular datasets with controlled bias distributions, enabling reproducible evaluation of fairness-aware FL algorithms.

Unlike standard FL benchmarks that often focus solely on data heterogeneity (e.g., non-IID label distributions), FeDa4Fair specifically addresses **fairness heterogeneity**. It simulates realistic scenarios where different clients (silos or devices) exhibit varying levels of bias against sensitive groups (e.g., based on race or gender).

## What This Library Does

FeDa4Fair provides a unified interface to:

1.  **Create Federated Datasets**: Easily partition tabular datasets into federated settings (Cross-Silo or Cross-Device).
2.  **Inject Controlled Bias**: systematically introduce **Attribute Skew** (varying population demographics) or **Value Skew** (varying conditional label distributions).
3.  **Group-Based Heterogeneity**: Partition clients into groups with distinct bias profiles sampled from **Truncated Normal Distributions**.
4.  **Evaluate Fairness**: Built-in tools to measure common fairness metrics like **Demographic Parity (DP)** and **Equalized Odds (EO)** both globally and at the client level.
4.  **Benchmark**: Access ready-to-use, pre-processed datasets with defined fairness characteristics.

## Supported Dataset Types

The library supports creating distinct types of federated datasets to model different real-world environments:

*   **Cross-Silo**: Simulates a setting with a small number of large clients (e.g., hospitals, banks, or states). 
    *   *Example*: Partitioning US Census data by State.
*   **Cross-Device**: Simulates a setting with a large number of small clients (e.g., mobile phones).
    *   *Example*: Partitioning a dataset into hundreds of small, non-overlapping subsets.
*   **Skewed Datasets**:
    *   **Attribute Skew**: The distribution of sensitive attributes varies across clients (e.g., some clients have mostly male users, others mostly female).
    *   **Value Skew**: The relationship between the sensitive attribute and the label varies across clients (e.g., historical bias affecting hiring decisions differs by region).

## Available Datasets

We provide 4 pre-configured benchmarking datasets derived from the ACS (American Community Survey) data, ready for immediate use:

1.  **Attribute-Silo**: A cross-silo dataset where the **attribute bias** (demographics) varies naturally across clients (States). ([Link](src/FeDa4Fair/data/cross_silo_attribute_final))
2.  **Attribute-Device**: A cross-device version where clients simulate devices with varying attribute distributions. ([Link](src/FeDa4Fair/data/cross_device_attribute_final))
3.  **Value-Silo**: A cross-silo dataset where **value bias** (correlation between race and outcome) varies across clients. ([Link](src/FeDa4Fair/data/cross_silo_value_final))
4.  **Value-Device**: A cross-device version with varying value bias. ([Link](src/FeDa4Fair/data/cross_device_value_final))

Additionally, FeDa4Fair has first-class support for:
*   **ACSIncome** (Folktables)
*   **ACSEmployment** (Folktables)
*   **Any Hugging Face Dataset** (see below)

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
uv run python examples/dutch.py
```

## Tutorial and Example 

A detailed example/tutorial on how to use the library can be found in [examples/acs_income.ipynb](examples/acs_income.ipynb).


## Run Formatting 

```bash
uv run ruff format
```

## Using Generic Datasets (Hugging Face & Local)

FeDa4Fair supports loading arbitrary datasets from Hugging Face Hub or local files.

### 1. Hugging Face Datasets

You can load any dataset available on Hugging Face Hub by specifying its name.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

fds = FairFederatedDataset(
    dataset="lucacorbucci/Dutch_Census",
    partitioners={"train": 10},  # Split train set into 10 partitions
    label_name="occupation_binary",
    sensitive_attributes=["sex_binary"],
    fairness_metric="DP"
)
fds.prepare()
```

See `examples/dutch.py` for a complete example.

### 2. Local Image Datasets

To use a local dataset (e.g., images), organize your data in a folder structure supported by Hugging Face `imagefolder` or `folder` builders, or simply point to the directory if it contains metadata.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

fds = FairFederatedDataset(
    dataset="imagefolder",
    data_dir="/path/to/local/data",
    partitioners={"train": 2},
    label_name="label",
    sensitive_attributes=["sensitive_attr"],
)
```

See `examples/celeba_example.py` for an example using a mock local dataset.