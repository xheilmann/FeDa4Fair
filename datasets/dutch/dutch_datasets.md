# Dutch Datasets

For the Dutch Census tabular dataset, we offer the following benchmarking datasets:

- **Value Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with value imbalance.
- **Attribute Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with feature imbalance
- **Standard Unfairness Datasets**: cross-device and cross-silo benchmarking datasets with all the clients imbalanced towards the same sensitive attribute value.

For each of these three categories, we provide both cross-device and cross-silo versions of the datasets with varying number of clients (50 in cross-silo and 150 in cross-device scenario) and varying degrees of imbalance/unfairness (mild, medium, and strong).

For each dataset that we create, we provide a script to reproduce the dataset creation process and an evaluation done on the created dataset measuring the Demographic Parity on the dataset and on the models trained on the datasets and the Equalized Odds on the models trained on the datasets.