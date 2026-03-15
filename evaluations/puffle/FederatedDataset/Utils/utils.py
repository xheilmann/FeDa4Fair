import numpy as np
import torch


class PartitionUtils:
    @staticmethod
    def create_splitted_dataset_from_tuple(
        splitted_indexes: np.array,
        dataset: tuple[torch.tensor, torch.tensor, torch.tensor],
    ):
        """
        This function partitions a dataset passed as parameter in N
        parts based on the splitted_indexes parameter.
        In particular, the dataset parameter is a tuple of three tensors:
        - indexes: the list of indexes
        - sensitive_attribute: the list of sensitive attributes
        - labels: the list of labels

        Args:
            splitted_indexes (np.array): The indexes of the dataset to be
                splitted
            dataset (tuple[torch.tensor, torch.tensor, torch.tensor]): The
                dataset to be splitted. It is a tuple of three tensors: indexes,
                sensitive_attribute, labels

        """
        images, sensitive_attribute, labels = dataset
        return [[images[indexes], sensitive_attribute[indexes], labels[indexes]] for indexes in splitted_indexes]
