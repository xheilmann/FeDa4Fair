from collections import Counter

import numpy as np
import torch


class NonIIDPartitionWithSensitiveFeature:
    @staticmethod
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        alpha=1000000,
    ) -> dict:
        if not alpha:
            msg = "Alpha must be a positive number"
            raise ValueError(msg)
        labels = dataset.targets
        sensitive_features = dataset.sensitive_features
        idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, lb in zip(idx, labels, strict=False):
            label_val = lb.item() if isinstance(lb, torch.Tensor) else lb
            if label_val not in index_per_label:
                index_per_label[label_val] = []
            index_per_label[label_val].append(index)

        list_labels = {item.item() if isinstance(item, torch.Tensor) else item for item in labels}
        list_sensitive_features = {
            item.item() if isinstance(item, torch.Tensor) else item for item in sensitive_features
        }
        labels_and_sensitive_feature = [(label, sf) for label in list_labels for sf in list_sensitive_features]

        if isinstance(labels, list):
            labels = np.array(labels)
        if isinstance(sensitive_features, list):
            sensitive_features = np.array(sensitive_features)
        to_be_sampled = []
        total_sampled = 0
        rng = np.random.default_rng()

        for label, sensitive_feature in labels_and_sensitive_feature:
            distribution = rng.dirichlet(num_partitions * [alpha], size=1)
            filtered_labels = labels[(labels == label) & (sensitive_features == sensitive_feature)]
            tmp_to_be_sampled = rng.choice(num_partitions, len(filtered_labels), p=distribution[0])
            total_sampled += len(tmp_to_be_sampled)
            to_be_sampled.append(Counter(tmp_to_be_sampled))

        if total_sampled != len(labels):
            msg = "Total sampled items does not match total labels"
            raise ValueError(msg)

        partitions_index = {f"node_{cluster_name}": [] for cluster_name in range(num_partitions)}
        total_samples = 0
        for (class_index, _), distribution_samples in zip(labels_and_sensitive_feature, to_be_sampled, strict=False):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[class_index][:samples]
                total_samples += samples
                index_per_label[class_index] = index_per_label[class_index][samples:]

        if total_samples != len(labels):
            msg = "Total samples items does not match total labels"
            raise ValueError(msg)

        partitions_labels = {
            cluster: [item.item() if isinstance(item, torch.Tensor) else item for item in labels[samples]]
            for cluster, samples in partitions_index.items()
        }

        return partitions_index, partitions_labels

    @staticmethod
    def do_partitioning_with_dataset_list(
        num_partitions: int,
        labels: list,
        sensitive_features: list,
        alpha=1000000,
    ) -> dict:
        dir_distributions = []
        if not alpha:
            msg = "Alpha must be a positive number"
            raise ValueError(msg)
        idx = torch.tensor(list(range(len(labels))))
        if isinstance(sensitive_features, torch.Tensor):
            sensitive_features = sensitive_features.numpy()

        index_per_label = {}
        for index, lb in zip(idx, labels, strict=False):
            label_val = lb.item() if isinstance(lb, torch.Tensor) else lb
            if label_val not in index_per_label:
                index_per_label[label_val] = []
            index_per_label[label_val].append(index)

        list_labels = {item.item() if isinstance(item, torch.Tensor) else item for item in labels}
        list_sensitive_features = {
            item.item() if isinstance(item, torch.Tensor) else item for item in sensitive_features
        }
        labels_and_sensitive_feature = [(label, sf) for label in list_labels for sf in list_sensitive_features]

        if isinstance(labels, list):
            labels = np.array(labels)
        if isinstance(sensitive_features, list):
            sensitive_features = np.array(sensitive_features)
        to_be_sampled = []
        total_sampled = 0
        rng = np.random.default_rng()

        for label, sensitive_feature in labels_and_sensitive_feature:
            distribution = rng.dirichlet(num_partitions * [alpha], size=1)
            dir_distributions.append((label, sensitive_feature, distribution))
            filtered_labels = labels[(labels == label) & (sensitive_features == sensitive_feature)]
            tmp_to_be_sampled = rng.choice(num_partitions, len(filtered_labels), p=distribution[0])
            total_sampled += len(tmp_to_be_sampled)
            to_be_sampled.append(Counter(tmp_to_be_sampled))

        if total_sampled != len(labels):
            msg = "Total sampled items does not match total labels"
            raise ValueError(msg)

        partitions_index = {f"node_{cluster_name}": [] for cluster_name in range(num_partitions)}
        total_samples = 0
        for (class_index, _), distribution_samples in zip(labels_and_sensitive_feature, to_be_sampled, strict=False):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[class_index][:samples]
                total_samples += samples
                index_per_label[class_index] = index_per_label[class_index][samples:]

        if total_samples != len(labels):
            msg = "Total samples items does not match total labels"
            raise ValueError(msg)

        partitions_labels = {
            cluster: [item.item() if isinstance(item, torch.Tensor) else item for item in labels[samples]]
            for cluster, samples in partitions_index.items()
        }

        partitions_index_list = []
        for node_name, indexes in partitions_index.items():
            partitions_index_list.append(indexes)
            node_labels = labels[indexes]
            node_sensitive_features = sensitive_features[indexes]
            node_labels_and_sensitive = zip(node_labels, node_sensitive_features, strict=False)
            print(f"Node {node_name} has: ", Counter(node_labels_and_sensitive))

        return (
            partitions_index,
            partitions_labels,
            partitions_index_list,
            dir_distributions,
        )
