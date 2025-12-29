import os
import sys
import unittest
from unittest.mock import MagicMock, call

import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from FeDa4Fair.metrics.fairness import _compute_fairness, compute_fairness
from FeDa4Fair.utils.data_utils import drop_data, flip_data


class TestFairnessComputation(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8], "sensitive": [0, 1, 0, 1], "label": [0, 1, 0, 1]}
        )

        # Mock partitioner
        self.mock_partition = MagicMock()
        self.mock_partition.to_pandas.return_value = self.df

        self.mock_partitioner = MagicMock()
        self.mock_partitioner.load_partition.return_value = self.mock_partition
        self.mock_partitioner.num_partitions = 2

    def test_internal_compute_fairness_eo(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]  # Some errors
        sf_data = pd.DataFrame({"sensitive": [0, 0, 1, 1]})

        # Test EO
        res = _compute_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sf_data=sf_data,
            fairness_metric="EO",
            sens_att="sensitive",
            size_unit="attribute",
        )
        self.assertIn("sensitive_EO", res.index)

    def test_internal_compute_fairness_attribute_value(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        sf_data = pd.DataFrame({"sensitive": [0, 0, 1, 1]})

        # Test attribute-value returns all pairwise diffs
        res = _compute_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sf_data=sf_data,
            fairness_metric="DP",
            sens_att="sensitive",
            size_unit="attribute-value",
        )
        # Pairs: 0_0, 0_1, 1_0, 1_1
        self.assertIn("0_1", res.index)
        self.assertEqual(len(res), 4)

    def test_compute_fairness_progress_callback(self):
        progress_callback = MagicMock()

        # Run compute_fairness with callback
        # We perform data bias check (model=None) to simplify
        compute_fairness(
            partitioner=self.mock_partitioner,
            partitioner_test=self.mock_partitioner,
            model=None,
            sens_att="sensitive",
            label_name="label",
            progress_callback=progress_callback,
        )

        # Assert callback was called for each partition (0 and 1)
        self.assertEqual(progress_callback.call_count, 2)
        progress_callback.assert_has_calls([call(0), call(1)])

    def test_compute_fairness_structure(self):
        # Test if output dataframe has expected structure
        result = compute_fairness(
            partitioner=self.mock_partitioner,
            partitioner_test=self.mock_partitioner,
            model=None,
            sens_att="sensitive",
            label_name="label",
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # 2 partitions
        # Check if index is named correctly
        self.assertEqual(result.index.name, "Partition ID")


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy data for utils tests
        # 10 rows
        # col1=1: 5 rows (4 True, 1 False)
        # col1=2: 5 rows
        self.df = pd.DataFrame(
            {
                "col1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "col2": ["a", "a", "b", "b", "c", "a", "a", "b", "b", "c"],
                "label": [True, True, True, True, False, True, True, False, False, False],
            }
        )

    def test_drop_data(self):
        # Drop 50% of rows where col1=1 and label=True
        # Matching rows: 4 rows (indices 0, 1, 2, 3)
        # Drop 50% -> 2 rows should be dropped.

        new_df = drop_data(df=self.df.copy(), percentage=0.5, column1="col1", value1=1, label_column="label")

        # Original length 10. Expected length 8.
        self.assertEqual(len(new_df), 8)

        # Verify that only matching rows were dropped
        # Non-matching rows should be untouched
        # col1=2 count should still be 5
        col1_2_val = 2
        expected_count = 5
        self.assertEqual(len(new_df[new_df["col1"] == col1_2_val]), expected_count)
        # col1=1 & label=False count should still be 1
        self.assertEqual(len(new_df[(new_df["col1"] == 1) & (~new_df["label"])]), 1)

    def test_flip_data(self):
        # Flip 50% of labels where col1=1 and label=True
        # Matching rows: 4 rows
        # Flip 50% -> 2 rows should have label changed to False.

        new_df = flip_data(df=self.df.copy(), percentage=0.5, column1="col1", value1=1, label_column="label")

        # Length should remain same
        self.assertEqual(len(new_df), 10)

        # Count of label=True for col1=1 should decrease by 2 (from 4 to 2)
        count_true = len(new_df[(new_df["col1"] == 1) & (new_df["label"])])
        self.assertEqual(count_true, 2)

        # Count of label=False for col1=1 should increase by 2 (from 1 to 3)
        count_false = len(new_df[(new_df["col1"] == 1) & (~new_df["label"])])
        self.assertEqual(count_false, 3)


if __name__ == "__main__":
    unittest.main()
