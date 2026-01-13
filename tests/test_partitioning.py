import unittest

import pandas as pd
import pytest

from datasets import Dataset
from FeDa4Fair.dataset.partitioning import RepresentativeDiversityPartitioner


class TestRepresentativeDiversityPartitioner(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset
        data = {
            "feature": range(100),
            "sensitive_1": ["A"] * 50 + ["B"] * 50,
            "sensitive_2": ["X", "Y"] * 50,
            "label": [0, 1] * 50,
        }
        self.df = pd.DataFrame(data)
        self.dataset = Dataset.from_pandas(self.df)

    def test_partitioning_single_attribute(self):
        # Test partitioning based on one sensitive attribute
        partitioner = RepresentativeDiversityPartitioner(num_partitions=2, partition_by="sensitive_1", seed=42)
        partitioner.dataset = self.dataset

        part_0 = partitioner.load_partition(0).to_pandas()
        part_1 = partitioner.load_partition(1).to_pandas()

        # Ensure correct types for checking
        if not isinstance(part_0, pd.DataFrame) or not isinstance(part_1, pd.DataFrame):
            msg = "Partition data is not a pandas DataFrame"
            raise TypeError(msg)

        # Check size balance (approximate)
        self.assertEqual(len(part_0), 50)
        self.assertEqual(len(part_1), 50)

        # Check if strata are preserved
        # Original: A=50, B=50. Each partition should have ~25 A and ~25 B
        self.assertEqual(len(part_0[part_0["sensitive_1"] == "A"]), 25)
        self.assertEqual(len(part_0[part_0["sensitive_1"] == "B"]), 25)

    def test_partitioning_intersectional(self):
        # Test partitioning based on two sensitive attributes
        partitioner = RepresentativeDiversityPartitioner(
            num_partitions=2, partition_by=["sensitive_1", "sensitive_2"], seed=42
        )
        partitioner.dataset = self.dataset

        part_0 = partitioner.load_partition(0).to_pandas()

        # Ensure correct types for checking
        if not isinstance(part_0, pd.DataFrame):
            msg = "Partition data is not a pandas DataFrame"
            raise TypeError(msg)

        # Groups: (A,X), (A,Y), (B,X), (B,Y). Each has 25 samples.
        # Each partition should get ~12 or 13 from each group.

        groups = self.df.groupby(["sensitive_1", "sensitive_2"]).size()
        part_groups = part_0.groupby(["sensitive_1", "sensitive_2"]).size()

        for group in groups.index:
            self.assertAlmostEqual(part_groups[group], groups[group] // 2, delta=1)

    def test_missing_column(self):
        partitioner = RepresentativeDiversityPartitioner(num_partitions=2, partition_by="non_existent_column")
        partitioner.dataset = self.dataset
        with pytest.raises(ValueError, match="Column .* not found"):
            partitioner.load_partition(0)

    def test_dataset_not_assigned(self):
        partitioner = RepresentativeDiversityPartitioner(num_partitions=2, partition_by="sensitive_1")
        with pytest.raises(ValueError, match="Dataset is not assigned"):
            partitioner.load_partition(0)
