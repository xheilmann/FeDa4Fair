import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from FeDa4Fair.visualization.plots import plot_comparison_label_distribution, plot_comparison_fairness_distribution


class TestPlotsExtended(unittest.TestCase):
    @patch("FeDa4Fair.visualization.plots.plt.subplots")
    @patch("FeDa4Fair.visualization.plots.plot_label_distributions")
    def test_plot_comparison_label_distribution(self, mock_pld, mock_subplots):
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        # For num_partitioners = 1, plt.subplots returns (fig, ax)
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_pld.return_value = (None, None, pd.DataFrame())

        partitioner = MagicMock()

        fig, axes, dfs = plot_comparison_label_distribution(partitioner_list=[partitioner], label_name="label")

        self.assertEqual(fig, mock_fig)
        self.assertEqual(len(axes), 1)
        mock_subplots.assert_called_once()

    @patch("FeDa4Fair.visualization.plots.plt.subplots")
    @patch("FeDa4Fair.visualization.plots.plot_fairness_distributions")
    def test_plot_comparison_fairness_distribution(self, mock_pfd, mock_subplots):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        # For num_partitioners = 1, plt.subplots returns (fig, ax)
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_pfd.return_value = (None, None, pd.DataFrame())

        partitioner = MagicMock()

        fig, axes, dfs = plot_comparison_fairness_distribution(
            partitioner_dict={"train": partitioner}, label_name="label", sens_att="sex"
        )

        self.assertEqual(fig, mock_fig)
        mock_subplots.assert_called_once()


if __name__ == "__main__":
    unittest.main()
