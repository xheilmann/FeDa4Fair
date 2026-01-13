import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from datasets import Dataset, DatasetDict

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Mock evaluation module BEFORE importing FairFederatedDataset
# This is necessary because evaluation imports xgboost which fails if libomp is missing
mock_evaluation = MagicMock()
sys.modules["FeDa4Fair.metrics.evaluation"] = mock_evaluation

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset


class TestFairFederatedDataset(unittest.TestCase):
    def setUp(self):
        self.mock_dataset = Dataset.from_dict(
            {"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8], "label": [0, 1, 0, 1]}
        )
        self.mock_dataset_dict = DatasetDict({"train": self.mock_dataset, "test": self.mock_dataset})

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_generic_dataset_loading(self, mock_load_dataset):
        # Setup mock return
        mock_load_dataset.return_value = self.mock_dataset_dict

        # Initialize
        fds = FairFederatedDataset(dataset="generic_dataset", partitioners={"train": 1, "test": 1}, label_name="label")

        # Trigger preparation
        fds.prepare()

        # Assertions
        mock_load_dataset.assert_called_once()
        # Compare keys and structure instead of direct object equality
        self.assertIsNotNone(fds._dataset)
        if fds._dataset is not None:
            self.assertEqual(list(fds._dataset.keys()), ["train", "test"])
            self.assertEqual(fds._dataset["train"].column_names, self.mock_dataset.column_names)
            self.assertEqual(fds._dataset["train"].num_rows, self.mock_dataset.num_rows)
        self.assertEqual(fds.label_column, "label")

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_single_split_loading(self, mock_load_dataset):
        # Setup mock return (single dataset, not dict)
        mock_load_dataset.return_value = self.mock_dataset

        # Initialize
        fds = FairFederatedDataset(
            dataset="generic_dataset", partitioners={"train": 1}, label_name="label", split="train"
        )

        # Trigger preparation
        fds.prepare()

        # Assertions
        # Should wrap in DatasetDict with key "train" (from split kwarg)
        self.assertIsInstance(fds._dataset, DatasetDict)
        self.assertIsNotNone(fds._dataset)
        if fds._dataset is not None:
            self.assertIn("train", fds._dataset)
            self.assertEqual(fds._dataset["train"].column_names, self.mock_dataset.column_names)
            self.assertEqual(fds._dataset["train"].num_rows, self.mock_dataset.num_rows)

    def test_acs_dataset_initialization(self):
        # Just testing init logic, not full loading which requires folktables
        # Provide partitioners as required
        fds = FairFederatedDataset(dataset="ACSIncome", partitioners={"train": 1})
        self.assertEqual(fds.label_column, "PINCP")
        self.assertIsNotNone(fds._states)
        if fds._states is not None:
            num_states = 51
            self.assertEqual(len(fds._states), num_states)  # 50 states + PR


if __name__ == "__main__":
    unittest.main()
