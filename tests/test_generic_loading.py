import unittest
from unittest.mock import patch

import pandas as pd

from datasets import Dataset, DatasetDict
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset


class TestGenericLoading(unittest.TestCase):
    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_merge_all_splits(self, mock_load_dataset):
        # Mock load_dataset to return a DatasetDict with multiple splits
        df_train = pd.DataFrame({"feature": [1, 2], "label": [0, 1], "sex": [0, 1]})
        df_test = pd.DataFrame({"feature": [3, 4], "label": [1, 0], "sex": [1, 0]})

        mock_load_dataset.return_value = DatasetDict(
            {"train": Dataset.from_pandas(df_train), "test": Dataset.from_pandas(df_test)}
        )

        # Initialize FairFederatedDataset with split="all"
        fds = FairFederatedDataset(
            dataset="dummy_hf_dataset",
            split="all",
            partitioners={"train": 1},
            label_name="label",
            sensitive_attributes=["sex"],
        )

        fds.prepare()

        # Verify load_dataset was called without 'split' argument
        # We need to check if 'split' is NOT in kwargs or is None if passed positionally
        # But our implementation pops it.
        # Check call args
        _args, kwargs = mock_load_dataset.call_args
        self.assertNotIn("split", kwargs)

        # Verify dataset structure
        self.assertIsNotNone(fds._dataset)
        if fds._dataset is None:
            return

        # Should have a single key "train" (default for merged)
        self.assertIn("train", fds._dataset)
        self.assertEqual(len(fds._dataset), 1)

        # Verify size (2+2 = 4)
        merged_ds = fds._dataset["train"]
        self.assertEqual(len(merged_ds), 4)


if __name__ == "__main__":
    unittest.main()
