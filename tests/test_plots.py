import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import matplotlib.pyplot as plt
from FeDa4Fair.visualization.plots import plot_fairness_distributions


class TestPlots(unittest.TestCase):
    @patch("FeDa4Fair.visualization.plots.compute_fairness")
    @patch("FeDa4Fair.visualization.plots._plot_heatmap")
    def test_plot_fairness_distributions(self, mock_plot_heatmap, mock_compute_fairness):
        # Setup mocks
        mock_df = pd.DataFrame({"sex_DP": [0.1, 0.2], "sex_val": ["0_1", "1_0"]})
        mock_compute_fairness.return_value = mock_df

        mock_ax = MagicMock()
        mock_fig = MagicMock(spec=plt.Figure)
        mock_ax.figure = mock_fig
        mock_plot_heatmap.return_value = mock_ax

        # Call function
        partitioner = MagicMock()
        fig, ax, df = plot_fairness_distributions(
            partitioner=partitioner, partitioner_test=partitioner, label_name="label", sens_att="sex"
        )

        # Assertions
        mock_compute_fairness.assert_called_once()
        mock_plot_heatmap.assert_called_once()
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        # sex_val should have been dropped
        self.assertIn("sex_DP", df.columns)
        self.assertNotIn("sex_val", df.columns)


if __name__ == "__main__":
    unittest.main()
