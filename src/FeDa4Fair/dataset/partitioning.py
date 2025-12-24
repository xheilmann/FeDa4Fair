from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner


class RepresentativeDiversityPartitioner(Partitioner):
    """
    Partitioner that ensures representative diversity by stratifying the dataset
    based on one or more sensitive attributes.

    Each partition will receive an equal proportion of samples from each subgroup
    defined by the sensitive attributes.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: Union[str, List[str]],
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by if isinstance(partition_by, list) else [partition_by]
        self._seed = seed
        self._indices_map: Optional[dict[int, List[int]]] = None

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def _determine_strata(self) -> None:
        """
        Identify subgroups and assign indices to partitions.
        """
        if self._dataset is None:
            raise ValueError("Dataset is not assigned to the partitioner.")

        # Convert to pandas for easier grouping
        df = self._dataset.to_pandas()

        # Ensure partition_by columns exist
        for col in self._partition_by:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")

        # Group by the sensitive attributes
        # We use a placeholder column to count or just iterate over groups
        groups = df.groupby(self._partition_by)

        partition_indices: dict[int, List[int]] = {i: [] for i in range(self._num_partitions)}

        rng = np.random.default_rng(self._seed)

        for _, group_df in groups:
            # Get indices of the current group
            indices = group_df.index.to_numpy()

            # Shuffle indices to ensure randomness within the stratum
            rng.shuffle(indices)

            # Split indices into num_partitions chunks
            # We use array_split to handle cases where len(indices) is not divisible by num_partitions
            chunks = np.array_split(indices, self._num_partitions)

            for i, chunk in enumerate(chunks):
                # If there are fewer chunks than partitions (very small group), loop around or handle?
                # array_split returns num_partitions arrays, some might be empty if len < num_partitions
                if i < self._num_partitions:
                    partition_indices[i].extend(chunk.tolist())

        self._indices_map = partition_indices

    def load_partition(self, partition_id: int) -> Dataset:
        if self._dataset is None:
            raise ValueError("Dataset is not assigned to the partitioner.")

        if self._indices_map is None:
            self._determine_strata()

        if partition_id not in self._indices_map:
            raise ValueError(f"Partition ID {partition_id} is out of range.")

        indices = self._indices_map[partition_id]
        return self._dataset.select(indices)
