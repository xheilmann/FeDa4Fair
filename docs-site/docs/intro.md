---
sidebar_position: 1
---

# Introduction

Welcome to **FeDa4Fair**, a comprehensive library designed to benchmark fairness in Federated Learning (FL).

## Overview

FeDa4Fair addresses a critical gap in FL research: the lack of standardized tools to evaluate how different types of data and fairness heterogeneity affect model performance. While many existing benchmarks focus on non-IID data distributions (e.g., label skew), FeDa4Fair introduces tools to simulate and evaluate **fairness heterogeneity**—where clients differ in their bias against sensitive groups.

With FeDa4Fair, you can:
*   **Create Federated Datasets**: Partition tabular data into Cross-Silo or Cross-Device settings.
*   **Inject Bias**: Systematically introduce **Attribute Skew** (demographic imbalance) or **Value Skew** (label correlation bias).
*   **Benchmark Fairness**: Evaluate models using standardized metrics like Demographic Parity and Equalized Odds, both globally and locally.

## Installation

FeDa4Fair is managed using [uv](https://github.com/astral-sh/uv).

### Prerequisites
*   Python 3.12+
*   `uv` package manager

### Setup

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies and Create Virtual Environment**:
    ```bash
    uv sync
    uv venv
    ```

3.  **Prepare Helper Directories**:
    ```bash
    mkdir -p src/FeDa4Fair/data_stats
    ```

## Quick Start

Here is a simple example of how to generate a dataset:

```bash
uv run python src/FeDa4Fair/main.py
```

This command runs the main script which demonstrates the generation of a federated dataset.

## Core Workflow

Using FeDa4Fair typically involves three steps:

1.  **Select a Dataset**: Use one of our pre-packaged ACS datasets (Income, Employment) or load your own from Hugging Face.
2.  **Define Partitioning**: Choose between **Cross-Silo** (e.g., state-based) or **Cross-Device** (many small clients) partitioning.
3.  **Configure Bias**: (Optional) Use the `modification_dict` to inject specific skews into client data.
4.  **Evaluate**: Use the provided evaluation tools to measure fairness metrics across your federated clients.

## Next Steps

*   Check out **[Creating Datasets](./CreatingDatasets.md)** to learn how to generate custom federated splits.
*   See **[Using Custom Datasets](./CustomDatasets.md)** for instructions on loading Hugging Face data.