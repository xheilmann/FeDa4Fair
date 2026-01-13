# Dutch Datasets

For the Dutch Census tabular dataset, we offer the following benchmarking datasets:

- **Value Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with value imbalance.
- **Attribute Imbalanced Datasets**: cross-device and cross-silo benchmarking datasets with feature imbalance
- **Standard Unfairness Datasets**: cross-device and cross-silo benchmarking datasets with all the clients imbalanced towards the same sensitive attribute value.

For each of these three categories, we provide both cross-device and cross-silo versions of the datasets with varying number of clients (50 in cross-silo and 150 in cross-device scenario) and varying degrees of imbalance/unfairness (mild, medium, and strong).

For each dataset that we create, we provide a script to reproduce the dataset creation process and an evaluation done on the created dataset measuring the Demographic Parity on the dataset and on the models trained on the datasets and the Equalized Odds on the models trained on the datasets.

In the attribute datasets, we use two sensitive attributes: "sex_binary" and "Marital_status", we want half of the clients that are unfair toward "sex_binary" while not being unfair toward "Marital_status" and vice versa for the second half of the clients.
In the value datasets, we use the sensitive attribute "sex_binary" and we want half of the clients to be unfair toward the value 1 and the other half toward the value 0. This can be measured using the Demographic Parity metric and checking which one of the two values of the sensitive attribute has the highest demographic parity. We can use the function violation_with_dataset to measure the unfairness and use the return value max_group to determine which value the client is unfair toward.