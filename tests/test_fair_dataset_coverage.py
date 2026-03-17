"""Tests for _clone_partitioner, to_json, and _create_and_cast_dataset in fair_dataset.py."""

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from flwr_datasets.partitioner import IidPartitioner

from datasets import ClassLabel, Dataset, DatasetDict
from FeDa4Fair.dataset.partitioning import RepresentativeDiversityPartitioner

# Mock evaluation module BEFORE importing FairFederatedDataset
# (avoids optional xgboost import crash on some systems)
mock_evaluation = MagicMock()
sys.modules.setdefault("FeDa4Fair.metrics.evaluation", mock_evaluation)

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset, _clone_partitioner


class TestClonePartitioner(unittest.TestCase):
    """Tests for the standalone _clone_partitioner helper."""

    def test_clones_simple_partitioner(self):
        """Should produce a new instance of the same type with the same args."""
        original = IidPartitioner(num_partitions=5)
        cloned = _clone_partitioner(original)

        self.assertIsInstance(cloned, IidPartitioner)
        self.assertIsNot(original, cloned)
        self.assertEqual(cloned.num_partitions, original.num_partitions)

    def test_cloned_instance_is_independent(self):
        """Assigning dataset to clone should not affect original."""
        original = IidPartitioner(num_partitions=3)
        cloned = _clone_partitioner(original)

        dummy_ds = Dataset.from_dict({"x": [1, 2, 3]})
        cloned.dataset = dummy_ds

        # Original should still have no dataset assigned
        self.assertFalse(hasattr(original, "dataset") and original.dataset is not None)

    def test_clones_representative_diversity_partitioner(self):
        """Should correctly clone the custom RepresentativeDiversityPartitioner."""
        original = RepresentativeDiversityPartitioner(num_partitions=4, partition_by="sex", seed=7)
        cloned = _clone_partitioner(original)

        self.assertIsInstance(cloned, RepresentativeDiversityPartitioner)
        self.assertIsNot(original, cloned)
        self.assertEqual(cloned.num_partitions, original.num_partitions)


class TestFairFederatedDatasetToJson(unittest.TestCase):
    """Tests for the to_json method."""

    def test_returns_valid_json_string(self):
        fds = FairFederatedDataset(dataset="ACSIncome", partitioners={"train": 1})
        result = fds.to_json()
        self.assertIsInstance(result, str)
        # Should parse without error
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)

    def test_json_contains_dataset_name(self):
        fds = FairFederatedDataset(dataset="ACSIncome", partitioners={"train": 1})
        result = fds.to_json()
        self.assertIn("ACSIncome", result)

    def test_json_contains_sensitive_attributes_when_set(self):
        fds = FairFederatedDataset(
            dataset="ACSIncome", partitioners={"train": 1}, sensitive_attributes=["SEX", "RAC1P"]
        )
        result = fds.to_json()
        self.assertIn("SEX", result)
        self.assertIn("RAC1P", result)

    def test_json_is_serializable_with_path_objects(self):
        """Path objects inside FDS should be serialized as strings, not crash."""
        fds = FairFederatedDataset(dataset="ACSIncome", partitioners={"train": 1})
        fds._path = Path("/some/path")
        result = fds.to_json()
        self.assertIsInstance(result, str)
        self.assertIn("/some/path", result)


class TestCreateAndCastDataset(unittest.TestCase):
    """Tests for _create_and_cast_dataset."""

    def _make_fds(self):
        return FairFederatedDataset(dataset="ACSIncome", partitioners={"train": 1})

    def test_converts_dataframe_to_dataset(self):
        fds = self._make_fds()
        df = pd.DataFrame({"feature": [1, 2, 3], "PINCP": [0, 1, 0]})
        result = fds._create_and_cast_dataset(df)
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 3)

    def test_casts_label_to_classlabel_when_possible(self):
        fds = self._make_fds()
        fds._label = "PINCP"
        df = pd.DataFrame({"feature": [1, 2, 3], "PINCP": [0, 1, 0]})
        result = fds._create_and_cast_dataset(df)
        # PINCP is the default label column for ACSIncome
        self.assertIsInstance(result.features["PINCP"], ClassLabel)

    def test_falls_back_gracefully_on_type_error(self):
        """If casting fails (e.g. non-numeric labels), should still return a Dataset."""
        fds = self._make_fds()
        df = pd.DataFrame({"feature": [1, 2], "PINCP": [{"nested": "obj"}, {"nested": "obj"}]})
        result = fds._create_and_cast_dataset(df)
        # Should not raise; just returns a plain Dataset
        self.assertIsInstance(result, Dataset)


class TestLoadPartitionWithSampleCap(unittest.TestCase):
    """Tests for load_partition behaviour when sample_cap is set."""

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_load_partition_respects_sample_cap(self, mock_load):
        """Loaded partition should be capped to sample_cap rows."""
        # 100-row dataset
        df = pd.DataFrame({"feature": range(100), "label": [0, 1] * 50})
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})

        fds = FairFederatedDataset(
            dataset="dummy",
            partitioners={"train": 1},
            label_name="label",
            sample_cap=20,
        )
        fds.prepare()

        partition = fds.load_partition(0, split="train")
        self.assertLessEqual(len(partition), 20)

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_load_partition_no_cap_returns_full_partition(self, mock_load):
        """Without a sample_cap, all rows in the partition should be accessible."""
        df = pd.DataFrame({"feature": range(50), "label": [0, 1] * 25})
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})

        fds = FairFederatedDataset(
            dataset="dummy",
            partitioners={"train": 1},
            label_name="label",
        )
        fds.prepare()

        partition = fds.load_partition(0, split="train")
        self.assertEqual(len(partition), 50)


class TestApplyModificationToPartition(unittest.TestCase):
    """Tests for _apply_modification_to_partition."""

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_drops_data_based_on_modification_dict(self, mock_load):
        """When modification_dict has drop_rate=1.0 for partition 0, all matching rows should be gone."""
        df = pd.DataFrame({"SEX": [1, 1, 2, 2], "label": [True, True, False, False]})
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})

        mod_dict = {
            0: {"SEX": {"drop_rate": 1.0, "flip_rate": 0, "value": 1, "attribute": None, "attribute_value": None}}
        }

        fds = FairFederatedDataset(
            dataset="dummy",
            partitioners={"train": 1},
            label_name="label",
            modification_dict=mod_dict,
        )
        fds.prepare()

        partition = fds.load_partition(0, split="train")
        partition_df = partition.to_pandas()
        # All rows where SEX=1 and label=True should be dropped
        self.assertEqual(len(partition_df[partition_df["SEX"] == 1]), 0)

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_no_modification_returns_unchanged_partition(self, mock_load):
        """Without a modification_dict, load_partition returns the original data."""
        df = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})

        fds = FairFederatedDataset(
            dataset="dummy",
            partitioners={"train": 1},
            label_name="label",
        )
        fds.prepare()

        partition = fds.load_partition(0, split="train")
        self.assertEqual(len(partition), 3)


class TestSaveDataset(unittest.TestCase):
    """Tests for save_dataset."""

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_save_dataset_creates_csv_files(self, mock_load):
        """save_dataset should write one CSV per partition per split."""
        df = pd.DataFrame({"feature": [1, 2], "label": [0, 1]})
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})

        fds = FairFederatedDataset(
            dataset="dummy",
            partitioners={"train": 1},
            label_name="label",
        )
        fds.prepare()

        with tempfile.TemporaryDirectory() as tmpdir:
            # save_dataset may warn about sensitive attributes; suppress for this test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fds.save_dataset(tmpdir)

            saved_files = [f.name for f in Path(tmpdir).iterdir()]
            # Should have written at least one CSV
            csv_files = [f for f in saved_files if f.endswith(".csv")]
            self.assertGreater(len(csv_files), 0)

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_save_dataset_csv_contains_correct_columns(self, mock_load):
        """Saved CSV should have the same columns as the original dataset."""
        df = pd.DataFrame({"feature": [10, 20], "label": [0, 1]})
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})

        fds = FairFederatedDataset(
            dataset="dummy",
            partitioners={"train": 1},
            label_name="label",
        )
        fds.prepare()

        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fds.save_dataset(tmpdir)

            csv_files = [f.name for f in Path(tmpdir).iterdir() if f.name.endswith(".csv")]
            self.assertTrue(csv_files)
            saved_df = pd.read_csv(Path(tmpdir) / csv_files[0])
            self.assertIn("feature", saved_df.columns)
            self.assertIn("label", saved_df.columns)


if __name__ == "__main__":
    unittest.main()
