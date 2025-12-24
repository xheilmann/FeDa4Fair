import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FeDa4Fair.metrics.evaluation as eval_mod
from FeDa4Fair.metrics.evaluation import evaluate_fairness, evaluate_models_on_datasets, local_client_fairness_plot

class TestEvaluationExtended(unittest.TestCase):
    def test_evaluate_fairness(self):
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        df = pd.DataFrame({"class0": [1, 2]}, index=[0, 1])
        df.index.name = "Partition ID"
        
        partitioner = MagicMock()
        partitioner.dataset.column_names = ["SEX"]
        partitioner.num_partitions = 10
        partitioner_dict = {"train": partitioner}
        
        with patch.object(eval_mod, "plot_comparison_label_distribution", return_value=(mock_fig, [mock_ax], [df])) as mock_pcld, \
             patch.object(eval_mod, "plot_comparison_fairness_distribution", return_value=(mock_fig, [mock_ax], [df])) as mock_pcfd, \
             patch.object(eval_mod, "merge_dataframes_with_names", return_value=pd.DataFrame()) as mock_merge, \
             patch.object(eval_mod, "Path") as mock_path, \
             patch.object(eval_mod, "pickle") as mock_pickle:
            
            evaluate_fairness(
                partitioner_dict=partitioner_dict,
                sens_columns=["SEX"]
            )
            
            mock_pcld.assert_called_once()
            mock_pcfd.assert_called_once()
            mock_fig.savefig.assert_called()

    def test_local_client_fairness_plot(self):
        df1 = pd.DataFrame({"Partition ID": [0, 1], "RAC1P_DP": [0.1, 0.2]})
        df2 = pd.DataFrame({"Partition ID": [0, 1], "RAC1P_DP": [0.15, 0.25]})
        
        fig = local_client_fairness_plot(df1, df2)
        
        self.assertIsInstance(fig, plt.Figure)

    @patch("FeDa4Fair.metrics.evaluation.evaluate_model")
    def test_evaluate_models_on_datasets(self, mock_evaluate_model):
        mock_evaluate_model.return_value = {"model": "LR", "accuracy": 0.8, "DP_SEX": 0.1}
        
        # Give it data with 2 samples to be safe
        datasets = [("State1", np.array([[1], [2]]), np.array([0, 1]), np.array([[1], [2]]), np.array([0, 1]), {"SEX": np.array([0, 1])})]
        
        # Use n_jobs=1 so that mocks are respected
        res_df, plt_obj = evaluate_models_on_datasets(datasets, n_jobs=1)
        
        self.assertIsInstance(res_df, pd.DataFrame)
        self.assertEqual(len(res_df), 1)
        self.assertEqual(res_df.iloc[0]["dataset"], "State1")
        self.assertEqual(plt_obj, plt)
