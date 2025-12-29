# FeDa4Fair

<p align="center">
  <img src="img/logo_no_background.png" width="200" alt="FeDa4Fair Logo">
</p>

**FeDa4Fair** is a library designed to facilitate the study of fairness in Federated Learning (FL). It allows researchers and practitioners to generate, manipulate, and benchmark datasets controlling bias distributions, enabling reproducible evaluation of fairness-aware FL algorithms.

Unlike standard FL benchmarks that often focus solely on data heterogeneity (e.g., non-IID label distributions), FeDa4Fair specifically addresses **fairness heterogeneity**. It simulates realistic scenarios where different clients (silos or devices) exhibit varying levels of bias against sensitive attributes (e.g., based on race or gender) or toward different values of the same sensitive attributes.

## What This Library Does

FeDa4Fair provides a unified interface to:

1.  **Create Federated Datasets**: Easily partition tabular datasets into federated settings (Cross-Silo or Cross-Device).
2.  **Inject Controlled Bias**: systematically introduce **Attribute Skew** (varying population demographics) or **Value Skew** (varying conditional label distributions).
3.  **Group-Based Heterogeneity**: Partition clients into groups with distinct bias profiles sampled from **Normal Distributions**.
4.  **Evaluate Fairness**: Built-in tools to measure common fairness metrics like **Demographic Parity (DP)** and **Equalized Odds (EO)** both globally and at the client level.
5.  **Benchmark**: Access ready-to-use, pre-processed datasets with defined fairness characteristics.

## Supported Dataset Types

The library supports creating distinct types of federated datasets to model different real-world environments:

*   **Cross-Silo**: Simulates a setting with a small number of large clients (e.g., hospitals, banks, or states). 
    *   *Example*: Partitioning US Census data by State.
*   **Cross-Device**: Simulates a setting with a large number of small clients (e.g., mobile phones).
    *   *Example*: Partitioning a dataset into hundreds of small, non-overlapping subsets.

## Available Datasets

We provide 4 pre-configured benchmarking datasets derived from the ACS (American Community Survey) data, ready for immediate use:

1.  **Attribute-Silo**: A cross-silo dataset where the **attribute bias** (demographics) varies naturally across clients (States). ([Link](src/FeDa4Fair/data/cross_silo_attribute_final))
2.  **Attribute-Device**: A cross-device version where clients simulate devices with varying attribute distributions. ([Link](src/FeDa4Fair/data/cross_device_attribute_final))
3.  **Value-Silo**: A cross-silo dataset where **value bias** (correlation between race and outcome) varies across clients. ([Link](src/FeDa4Fair/data/cross_silo_value_final))
4.  **Value-Device**: A cross-device version with varying value bias. ([Link](src/FeDa4Fair/data/cross_device_value_final))

Additionally, FeDa4Fair has first-class support for:
*   **ACSIncome** (Folktables)
*   **ACSEmployment** (Folktables)
*   **Dutch Census** (Hugging Face)
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

## Tutorial and Example 

A detailed example/tutorial on how to use the library can be found in the example folder:

* We provide an example with the [ACS Income](examples/acs_income.ipynb) dataset
* We provide an example with the [Dutch Census](examples/dutch.ipynb) dataset
* We provide an example with an image dataset, [CelebA](examples/celeba.ipynb), using data from Hugging Face Hub.


## How to use the library

FeDa4Fair supports loading arbitrary datasets from Hugging Face Hub or local files.

### 1. Hugging Face Datasets

You can load any dataset available on Hugging Face Hub by specifying its name.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

fds = FairFederatedDataset(
    dataset="flwrlabs/celeba",
    split="all",
    partitioners={"train": num_total_clients},
    label_name="Smiling",
    sensitive_attributes=["Male"],
    fairness_metric="DP",
    modification_dict=mod_dict,
)

fds.prepare()
```

In this case, we load the CelebA dataset, partition it into `num_total_clients`, and set "Smiling" as the label and "Male" as the sensitive attribute. The `modification_dict` can be used to specify bias injection parameters.
An example of `modification_dict` for attribute skew could be:

```python
group_configs = [
    {
        "group_id": "Group A (Unfair to Female)",
        "num_clients": 75,
        "sensitive_attr": "Male",
        "sensitive_value": "false",  # Female
        "drop_mean": 0.2,
        "drop_std": 0.05,
        "flip_mean": 0.1,
        "flip_std": 0.02,
    },
    {
        "group_id": "Group B (Unfair to Male)",
        "num_clients": 75,
        "sensitive_attr": "Male",
        "sensitive_value": "true",  # Male
        "drop_mean": 0.3,
        "drop_std": 0.05,
        "flip_mean": 0.2,
        "flip_std": 0.02,
    },
]
mod_dict = generate_bias_by_groups(num_total_clients=num_total_clients, group_configs=group_configs)
```

This configuration creates two groups of clients with different bias profiles regarding the "Male" attribute. 

By specifying "all" in the `split` parameter, we ensure that all the dataset splits (train, validation, test) available on Hugging face are downloaded and merged into a single dataset before partitioning. Later, we will split the data into train, validation, and test sets at the client level to support federated learning scenarios.

See `examples/celeba.ipynb` for a complete example. 


### 2. Local Datasets

To use a local dataset, organize your data in a folder structure supported by Hugging Face `imagefolder` or `folder` builders, or simply point to the directory if it contains metadata.

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

ffds = FairFederatedDataset(
    dataset="ACSIncome",
    states=["CT", "AK"],
    partitioners={"CT": 5, "AK": 2},
    fl_setting=None,
    fairness_metric="DP",
    fairness_level="attribute",
)

```

See `examples/acs_income.ipynb` for an example using ACSIncome locally downloaded.

## How to use our dashboard

FeDa4Fair includes a dashboard for easily split dataset into clients and visualize fairness metrics.
To launch the dashboard, run the following command from the dashboard folder:

```
uv run streamlit run app.py
```

This will start a local server, and you can access the dashboard through your web browser at `http://localhost:8501`.


<p align="center">
  <img src="img/dashboard_1.png" alt="Dashboard screenshot">
</p>

Through the dashboard, you can select the dataset, define the federated setting (Cross-Silo or Cross-Device), specify bias injection parameters, and visualize the resulting fairness metrics across clients.

<p align="center">
  <img src="img/dashboard_2.png" alt="Dashboard screenshot">
</p>

## Citation

If you use FeDa4Fair in your research, please cite the following paper:

```
@misc{heilmann2025feda4fairclientlevelfederateddatasets,
      title={FeDa4Fair: Client-Level Federated Datasets for Fairness Evaluation}, 
      author={Xenia Heilmann and Luca Corbucci and Mattia Cerrato and Anna Monreale},
      year={2025},
      eprint={2506.21095},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.21095}, 
}
```