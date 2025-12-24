import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datasets import Dataset, DatasetDict
from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

class TestFairDatasetExtended(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "SEX": [1, 2, 1, 2],
            "label": [0, 1, 0, 1]
        })
        self.ds = Dataset.from_pandas(self.df)

    @patch("FeDa4Fair.dataset.fair_dataset.load_dataset")
    def test_modification_application(self, mock_load):
        mock_load.return_value = DatasetDict({"train": self.ds})
        
        # Define modification: drop 100% of SEX=1 where label is True
        # In our data, SEX=1 has label=0. So nothing should be dropped?
        # Wait, drop_data drops where label is True.
        # Let's change data to have SEX=1 and label=True
        df = pd.DataFrame({
            "SEX": [1, 1, 2, 2],
            "label": [True, True, False, False]
        })
        ds = Dataset.from_pandas(df)
        mock_load.return_value = DatasetDict({"train": ds})
        
        mod_dict = {
            "train": {
                "SEX": {
                    "drop_rate": 1.0,
                    "value": 1
                }
            }
        }
        
        fds = FairFederatedDataset(
            dataset="generic",
            partitioners={"train": 1},
            label_name="label",
            modification_dict=mod_dict
        )
        fds.prepare()
        
        # Check if modifications were applied to fds._dataset
        train_ds = fds._dataset["train"]
        # Originally 2 rows with SEX=1 and label=True. Both should be dropped.
        # Remaining rows should be the ones with SEX=2.
        self.assertEqual(len(train_ds), 2)
        self.assertEqual(train_ds[0]["SEX"], 2)

    def test_get_default_us_states(self):
        fds = FairFederatedDataset(dataset="ACSIncome", partitioners={"train": 1})
        states = fds._get_default_us_states()
        self.assertIn("CA", states)
        self.assertIn("PR", states)
        self.assertEqual(len(states), 51)

    @patch("FeDa4Fair.dataset.fair_dataset.FairFederatedDataset.load_acs_raw_data")
    @patch("FeDa4Fair.dataset.fair_dataset.Divider")
    def test_split_into_train_val_test(self, mock_divider, mock_load_acs):
        mock_load_acs.return_value = {"CA": self.df}
        # Mock Divider behavior
        mock_divider_instance = MagicMock()
        mock_divider.return_value = mock_divider_instance
        mock_divider_instance.return_value = DatasetDict({
            "CA_train": self.ds, "CA_val": self.ds, "CA_test": self.ds
        })
        
        fds = FairFederatedDataset(
            dataset="ACSIncome",
            states=["CA"],
            partitioners={"CA": 1},
            fl_setting="cross-silo",
            sensitive_attributes=["SEX"]
        )
        fds.prepare()
        
        self.assertIn("CA_train", fds._dataset)
        self.assertIn("CA_val", fds._dataset)
        self.assertIn("CA_test", fds._dataset)

if __name__ == "__main__":
    unittest.main()
