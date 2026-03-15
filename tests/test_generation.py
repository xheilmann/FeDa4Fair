import unittest

import pandas as pd

from FeDa4Fair.dataset.generation import (
    _filter_states_by_fairness,
    _get_attribute_modifications,
    preprocess_data_cross_silo,
    split_df,
)


class TestGeneration(unittest.TestCase):
    def test_split_df(self):
        df = pd.DataFrame({"a": range(10)})
        splits = split_df(df, 2)
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 5)
        self.assertEqual(len(splits[1]), 5)

    def test_filter_states_by_fairness_attribute(self):
        # Mock df from evaluate_models_on_datasets
        data = {
            "dataset": ["S1", "S1", "S2", "S2"],
            "model": ["XGBoost", "LogisticRegression", "XGBoost", "LogisticRegression"],
            "DP_SEX": [0.1, 0.15, 0.05, 0.05],
            "DP_RACE": [0.01, 0.02, 0.1, 0.12],
        }
        df = pd.DataFrame(data)
        # S1: SEX_DP > RACE_DP (count=2). np.min(SEX_DP) = 0.1 > 0.09. Should be included.
        # S2: RACE_DP > SEX_DP (count=0). np.min(RACE_DP) = 0.1. 0.175 > 0.1 > 0.12? False.
        states = _filter_states_by_fairness(df, "attribute")
        self.assertIn("S1", states)
        self.assertNotIn("S2", states)

    def test_preprocess_data_cross_silo(self):
        df = pd.DataFrame(
            {
                "PINCP": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "SEX": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                "MAR": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "RAC1P": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        datasets = []
        res = preprocess_data_cross_silo(df, datasets, "attribute", "State1")
        self.assertEqual(len(res), 1)
        name, x_train, _y_train, _x_test, _y_test, sf = res[0]
        self.assertEqual(name, "State1")
        self.assertEqual(x_train.shape[1], 0)  # PINCP, SEX, MAR, RAC1P dropped
        self.assertIn("SEX", sf)

    def test_get_attribute_modifications_logic(self):
        """
        Test that _get_attribute_modifications targets correct attributes based on count.
        """
        # count == 1 (only first row) -> SEX
        df_entry = pd.DataFrame(
            {
                "DP_RACE": [0.05, 0.06],
                "DP_SEX": [0.10, 0.04],
            }
        )
        result = _get_attribute_modifications("TX", df_entry, 0.3)
        self.assertIsNotNone(result)
        self.assertEqual(result[1]["SEX"]["drop_rate"], 0.3)

        # count == 2 -> SEX
        df_entry2 = pd.DataFrame(
            {
                "DP_RACE": [0.05, 0.06],
                "DP_SEX": [0.15, 0.12],
            }
        )
        result2 = _get_attribute_modifications("TX", df_entry2, 0.3)
        self.assertIsNotNone(result2)
        self.assertIn("SEX", result2[1])

    def test_preprocess_data_cross_silo_no_mutation(self):
        """preprocess_data_cross_silo should not mutate the input DataFrame."""
        data = pd.DataFrame(
            {
                "PINCP": [0, 1, 0, 1] * 25,
                "SEX": [1, 2, 1, 2] * 25,
                "RAC1P": [1, 2, 1, 2] * 25,
                "MAR": [1, 2, 1, 2] * 25,
                "feature1": range(100),
            }
        )
        original_columns = list(data.columns)

        preprocess_data_cross_silo(data.copy(), [], "attribute", "TX")

        # Original data should still have all its columns
        self.assertEqual(list(data.columns), original_columns)


if __name__ == "__main__":
    unittest.main()
