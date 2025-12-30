import numpy as np
import pandas as pd
from flwr_datasets.partitioner import Partitioner

from datasets import Dataset


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
        partition_by: str | list[str],
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by if isinstance(partition_by, list) else [partition_by]
        self._seed = seed
        self._indices_map: dict[int, list[int]] | None = None

    @property
    def num_partitions(self) -> int:
        """Return the number of partitions."""
        return self._num_partitions

    @property
    def partition_by(self) -> list[str]:
        """Return the columns used for partitioning."""
        return self._partition_by

    @property
    def seed(self) -> int | None:
        """Return the random seed."""
        return self._seed

    def _determine_strata(self) -> None:
        """
        Identify subgroups and assign indices to partitions.
        """
        if self._dataset is None:
            msg = "Dataset is not assigned to the partitioner."
            raise ValueError(msg)

        # Convert to pandas for easier grouping
        df = self._dataset.to_pandas()
        if not isinstance(df, pd.DataFrame):
            msg = "Dataset must be convertible to a pandas DataFrame."
            raise TypeError(msg)

        # Ensure partition_by columns exist
        for col in self._partition_by:
            if col not in df.columns:
                msg = f"Column '{col}' not found in dataset."
                raise ValueError(msg)

        # Group by the sensitive attributes
        # We use a placeholder column to count or just iterate over groups
        groups = df.groupby(self._partition_by)

        partition_indices: dict[int, list[int]] = {i: [] for i in range(self._num_partitions)}

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
                if i < self._num_partitions:
                    partition_indices[i].extend(chunk.tolist())

        self._indices_map = partition_indices

    def load_partition(self, partition_id: int) -> Dataset:
        if self._dataset is None:
            msg = "Dataset is not assigned to the partitioner."
            raise ValueError(msg)

        if self._indices_map is None:
            self._determine_strata()

        # Check again to satisfy type checker
        if self._indices_map is None:
            msg = "Indices map could not be created."
            raise ValueError(msg)

        if partition_id not in self._indices_map:
            msg = f"Partition ID {partition_id} is out of range."
            raise ValueError(msg)

        indices = self._indices_map[partition_id]
        return self._dataset.select(indices)
