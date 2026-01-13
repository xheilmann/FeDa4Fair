# CelabA Datasets

For the CelabA image dataset, we offer the following benchmarking datasets:

- **Value Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with value imbalance.
- **Attribute Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with feature imbalance

For each of these two categories, we provide both cross-device and cross-silo versions of the datasets with varying number of clients (50 in cross-silo and 150 in cross-device scenario).

For each dataset that we create, we provide a script to reproduce the dataset creation process and an evaluation done on the created dataset measuring the Demographic Parity on the dataset. Note: Due to the computational cost of training models on image data, these scripts only measure Demographic Parity on the dataset itself, not on trained models.

For the attribute imbalanced datasets, we focus on the "Male" sensitive attribute. We want half of the clients to be unfair for the "Male" attribute (inject biases) and fair (mitigate the unfairness) for the attribute "hair color" while the other half are fair for "Male" (mitigate the unfairness) but they are unfair for the attribute "hair color" (inject unfairness).

For the hair color attribute, create a pre-processing function that adds a new column "hair_color" to the CelebA dataset. This column indicates the hair color. The original csv file of celeba contains multiple columns for different hair colors (e.g., "Black_Hair", "Blond_Hair", "Brown_Hair", etc.). The pre-processing function combines these columns into a single "hair_color" column, based on what are the hair colors present, we group them into two categories: "Dark" (Black, Brown) and "Light" (Blond, Gray, etc.).

For the value imbalanced datasets, we use a different pre-processing. Hair color in this case is a column that contains multiple values (the ones that are in the original Csv file for celeba). Then, we create half of the clients that are unfair for the attribute value that is less present in the dataset (count the one that is less present) and the other half that is unfair for the attribute value that is more present in the dataset. Create the splits in a way that this can happen. 

## Usage

To generate the datasets and run the evaluation, execute the corresponding scripts:

### Cross-Device (150 clients)


- **Attribute Imbalance**:
  ```bash
  uv run python datasets/celeba/create_cross_device_attribute.py
  ```
- **Value Imbalance**:
  ```bash
  uv run python datasets/celeba/create_cross_device_value.py
  ```

### Cross-Silo (50 clients)


- **Attribute Imbalance**:
  ```bash
  python datasets/celeba/create_cross_silo_attribute.py
  ```
- **Value Imbalance**:
  ```bash
  python datasets/celeba/create_cross_silo_value.py
  ```

## Results

The scripts calculate the average Demographic Parity (DP) difference across clients. The results are saved in CSV files within the respective dataset directories (e.g., `datasets/celeba/cross_device_standard/medium_evaluation.csv`).

Target DP levels is approximemately 0.30

(Run the scripts to generate the exact values)

## Plots

I need plots similar to the ones used for Dutch, for attribute datasets I need a plot with showing the unfairness of each client for the two attributes ( Male and hair color) and for value datasets a plot showing the unfairness of each client for the different values of hair color (show the highest one for each client).