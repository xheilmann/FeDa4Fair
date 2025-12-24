import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import json
from pathlib import Path
from FeDa4Fair.utils.reporting import compute_sensitive_attr_proportions, create_new_datasheet


class TestReporting(unittest.TestCase):
    def setUp(self):
        # Mock FairFederatedDataset
        self.mock_fds = MagicMock()
        self.mock_fds._dataset_prepared = True
        self.mock_fds._sensitive_attributes = ["SEX"]
        self.mock_fds.label_column = "label"

        # Mock dataset dict
        self.df = pd.DataFrame({"SEX": [1, 1, 2, 2], "label": [True, False, True, False]})
        self.mock_ds = MagicMock()
        self.mock_ds.column_names = ["SEX", "label"]
        self.mock_ds.__len__.return_value = 4
        self.mock_ds.__iter__.return_value = self.df.to_dict("records")

        self.mock_fds._dataset = {"train": self.mock_ds}

        # Mock partitioners
        self.mock_partitioner = MagicMock()
        self.mock_partitioner.num_partitions = 1
        self.mock_partitioner.load_partition.return_value = self.df
        self.mock_fds._partitioners = {"train": self.mock_partitioner}
        self.mock_fds._modification_dict = {"train": {"SEX": {"drop_rate": 0.1}}}

    def test_compute_sensitive_attr_proportions(self):
        stats = compute_sensitive_attr_proportions(self.mock_fds, sensitive_attrs=["SEX"])

        self.assertIn("overall", stats)
        self.assertIn("splits", stats)
        self.assertIn("partitions", stats)

        # SEX=1 is 50%, SEX=2 is 50%
        self.assertEqual(stats["overall"]["SEX"][1], 0.5)
        self.assertEqual(stats["overall"]["SEX"][2], 0.5)

    @patch("FeDa4Fair.utils.reporting.prep_info_dict")
    @patch("FeDa4Fair.utils.reporting.SOURCE_FILE")
    def test_create_new_datasheet(self, mock_source_file, mock_prep_info):
        mock_prep_info.return_value = {"income": "KEEP", "name": "Test"}
        # Mock template content with tags
        mock_source_file.read_text.return_value = "[tag:name]Dataset[/tag] [tag:income]Income Info[/tag]"

        # Mock dataset.to_json()
        self.mock_fds.to_json.return_value = json.dumps(
            {"_dataset_name": "ACSIncome", "_year": "2018", "_horizon": "1-Year", "_sensitive_attributes": ["SEX"]}
        )

        # Run creation
        with patch("pathlib.Path.write_text") as mock_write:
            with patch("pathlib.Path.mkdir"):
                create_new_datasheet("dummy_path", self.mock_fds)

                # Check if write_text was called
                mock_write.assert_called_once()
                # Content should contain replaced tags
                # name replacement: ACSIncomeFeDa4Fair...
                # income replacement: "Income Info" (since it was "KEEP")
                written_content = mock_write.call_args[0][0]
                self.assertIn("ACSIncomeFeDa4Fair", written_content)
                self.assertIn("Income Info", written_content)


if __name__ == "__main__":
    unittest.main()
