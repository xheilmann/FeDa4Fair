import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from FeDa4Fair.metrics import evaluation
from FeDa4Fair.metrics.evaluation import evaluate_model, merge_dataframes_with_names


class TestEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        # Mock return value
        mock_series = pd.Series([0.1, "0_1"], index=["SEX_DP", "SEX_val"])

        # Data
        x_train, y_train = np.array([[1], [2]]), np.array([0, 1])
        x_test, y_test = np.array([[1], [2]]), np.array([0, 1])
        sf_data = {"SEX": np.array([0, 1])}

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])

        # Patch using patch.object
        with patch.object(evaluation, "_compute_fairness", return_value=mock_series) as mock_cf:
            res = evaluate_model(
                model_name="LR",
                model=mock_model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                fairness_metric="DP",
                sf_data=sf_data,
                fairness_level="attribute",
            )

            print(f"Mock called: {mock_cf.called}")
            self.assertTrue(mock_cf.called)
            self.assertEqual(res["DP_SEX"], 0.1)

    def test_merge_dataframes_with_names(self):
        df1 = pd.DataFrame({"val": [1]})
        df2 = pd.DataFrame({"val": [2]})
        names = ["A", "B"]

        merged = merge_dataframes_with_names([df1, df2], names)

        self.assertEqual(len(merged), 2)
        self.assertIn("state", merged.columns)
        self.assertEqual(merged.iloc[0]["state"], "A")
        self.assertEqual(merged.iloc[1]["state"], "B")


if __name__ == "__main__":
    unittest.main()
