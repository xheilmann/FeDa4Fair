# CelabA Datasets

For the CelabA image dataset, we offer the following benchmarking datasets:

- **Value Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with value imbalance.
- **Attribute Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with feature imbalance
- **Standard Unfairness Datasets**: cross-device and cross-silo benchmarking datasets with all the clients imbalanced towards the same sensitive attribute value.

For each of these three categories, we provide both cross-device and cross-silo versions of the datasets with varying number of clients (50 in cross-silo and 150 in cross-device scenario) and varying degrees of imbalance/unfairness (mild, medium, and strong).

For each dataset that we create, we provide a script to reproduce the dataset creation process and an evaluation done on the created dataset measuring the Demographic Parity on the dataset. Note: Due to the computational cost of training models on image data, these scripts only measure Demographic Parity on the dataset itself, not on trained models.

## Usage

To generate the datasets and run the evaluation, execute the corresponding scripts:

### Cross-Device (150 clients)

- **Standard Unfairness**:
  ```bash
  python datasets/celeba/create_cross_device_standard.py
  ```
- **Attribute Imbalance**:
  ```bash
  python datasets/celeba/create_cross_device_attribute.py
  ```
- **Value Imbalance**:
  ```bash
  python datasets/celeba/create_cross_device_value.py
  ```

### Cross-Silo (50 clients)

- **Standard Unfairness**:
  ```bash
  python datasets/celeba/create_cross_silo_standard.py
  ```
- **Attribute Imbalance**:
  ```bash
  python datasets/celeba/create_cross_silo_attribute.py
  ```
- **Value Imbalance**:
  ```bash
  python datasets/celeba/create_cross_silo_value.py
  ```

## Results

The scripts calculate the average Demographic Parity (DP) difference across clients. The results are saved in CSV files within the respective dataset directories (e.g., `datasets/celeba/cross_device_standard/mild_evaluation.csv`).

Target DP levels are approximately:
- Mild: ~0.15
- Medium: ~0.25
- Strong: ~0.35

(Run the scripts to generate the exact values)
