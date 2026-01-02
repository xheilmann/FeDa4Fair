---
sidebar_position: 4
---

# Benchmarking Datasets

FeDa4Fair provides four pre-packaged datasets designed to benchmark fairness-aware Federated Learning algorithms. These datasets are derived from the **ACS PUMS (American Community Survey Public Use Microdata Sample)** and are specifically engineered to exhibit different types of fairness heterogeneity (bias) across clients.

## Overview

The datasets cover two main federated settings and two types of bias:

| Dataset Name | Setting | Bias Type | Description |
| :--- | :--- | :--- | :--- |
| **Attribute-Silo** | Cross-Silo | Attribute Skew | Clients (States) have varying distributions of sensitive attributes (e.g., some have very few female samples). |
| **Value-Silo** | Cross-Silo | Value Skew | Clients have varying distributions of specific values within a sensitive attribute (e.g., different racial groups are underrepresented in different states). |
| **Attribute-Device** | Cross-Device | Attribute Skew | Similar to Attribute-Silo but partitioned into many small clients to simulate devices. |
| **Value-Device** | Cross-Device | Value Skew | Similar to Value-Silo but partitioned into many small clients. |

## 1. Cross-Silo Attribute Bias (`cross_silo_attribute_final`)

This dataset simulates a cross-silo scenario (e.g., a consortium of hospitals or regional centers) where the demographic composition varies significantly between clients.

*   **Base Data**: ACS Income task (predicting income > $50k).
*   **Clients**: US States.
*   **Total Rows**: ~1,626,658
*   **Unfairness Injection**:
    Specific sensitive groups were downsampled in certain states to create **Attribute Skew**. This means the *proportion* of the sensitive group varies across clients.

    **Modification Table (Examples):**

    | Client (State) | Attribute | Value Dropped | Drop Rate |
    | :--- | :--- | :--- | :--- |
    | **MO, PR** | SEX | 2 (Female) | **0.6** (60% dropped) |
    | **MI, VA, TN, OH** | SEX | 2 (Female) | **0.5** |
    | **PA, WV, AR, KS, TX, OK, ID** | SEX | 2 (Female) | **0.4** |
    | **OR, DE** | RAC1P | 2 (Non-White) | **0.4** |
    | **IL, WA, NH** | SEX | 2 (Female) | **0.3** |
    | ... | ... | ... | ... |

    *Note: A drop rate of 0.6 means 60% of the instances belonging to that group were removed from that client's data.*

## 2. Cross-Silo Value Bias (`cross_silo_value_final`)

This dataset simulates a scenario where the bias manifests in the *representation of specific subgroups* within a sensitive attribute.

*   **Base Data**: ACS Income task.
*   **Clients**: US States.
*   **Total Rows**: ~740,932
*   **Unfairness Injection**:
    Specific racial groups (values of `RAC1P`) were downsampled in specific states.

    **Modification Table (Examples):**

    | Client (State) | Attribute | Value Dropped | Drop Rate |
    | :--- | :--- | :--- | :--- |
    | **PR** | RAC1P | 4 | **0.6** |
    | **AK, MS** | RAC1P | 4 | **0.5** |
    | **DE, NE** | RAC1P | 4 | **0.3** |
    | **LA** | RAC1P | 5 | **0.3** |
    | **AR, OR** | RAC1P | 4 / 2 | **0.2** |
    | **MN, WV** | RAC1P | 5 | **0.2** |
    | **AZ** | RAC1P | 5 | **0.1** |
    | **OH** | RAC1P | 4 | **0.1** |

## 3. Cross-Device Attribute Bias (`cross_device_attribute_final`)

Designed for cross-device FL (e.g., mobile phones), this dataset partitions the data into many smaller clients.

*   **Base Data**: ACS Income task.
*   **Clients**: Hundreds of small partitions derived from state data.
*   **Total Rows**: ~740,932
*   **Unfairness Injection**:
    Follows a similar pattern to the Attribute-Silo dataset, injecting attribute skew by dropping specific demographic groups in the underlying state data before partitioning into devices.

## 4. Cross-Device Value Bias (`cross_device_value_final`)

Designed for cross-device FL with value skew.

*   **Base Data**: ACS Income task.
*   **Clients**: Hundreds of small partitions.
*   **Total Rows**: ~1,144,252
*   **Unfairness Injection**:
    Follows the Value-Silo pattern, creating imbalances in specific racial subgroups across the simulated device population.

## Usage

To use these datasets, simply reference them by name when initializing `FairFederatedDataset`:

```python
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

# Load the Cross-Silo Attribute Bias dataset
fds = FairFederatedDataset(
    dataset="cross_silo_attribute_final", # Maps to the internal path
    partitioners={"train": 10}, # Example partitioner
    fairness_metric="DP"
)
fds.prepare()
```
