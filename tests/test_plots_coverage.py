"""Tests for plot_multi_attribute_fairness and _prepare_value_plot_data in plots.py."""

import contextlib
import unittest
from unittest.mock import MagicMock, patch

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt

from FeDa4Fair.visualization.plots import _prepare_value_plot_data, plot_multi_attribute_fairness


def _make_mock_partitioner(df: pd.DataFrame, num_partitions: int = 3) -> MagicMock:
    """Return a mock partitioner whose load_partition always yields df."""
    partition = MagicMock()
    partition.to_pandas.return_value = df
    partition.__len__ = MagicMock(return_value=len(df))

    partitioner = MagicMock()
    partitioner.load_partition.return_value = partition
    partitioner.num_partitions = num_partitions
    return partitioner


def _make_combined_df(n_partitions: int = 3) -> pd.DataFrame:
    """Build a combined_df with SEX and RAC1P DP columns for plotting tests."""
    rng = np.random.default_rng(0)
    data = {
        "SEX_DP": rng.uniform(0, 0.3, n_partitions),
        "SEX_val": [f"{i}_{i + 1}" for i in range(n_partitions)],
        "RAC1P_DP": rng.uniform(0, 0.3, n_partitions),
        "RAC1P_val": [f"{i}_{i + 1}" for i in range(n_partitions)],
        "Sample Count": [100] * n_partitions,
    }
    df = pd.DataFrame(data)
    df.index.name = "Partition ID"
    return df


class TestPrepareValuePlotData(unittest.TestCase):
    """Tests for the internal _prepare_value_plot_data helper."""

    def setUp(self):
        # Simulate a combined_df that has attribute-value level fairness columns
        # e.g. SEX_1_2 means the pairwise DP between SEX=1 and SEX=2
        n = 4
        rng = np.random.default_rng(1)
        self.combined_df = pd.DataFrame(
            {
                "SEX_1_2": rng.uniform(0, 0.3, n),
                "SEX_2_1": rng.uniform(0, 0.3, n),
                "RAC1P_1_2": rng.uniform(0, 0.2, n),
                "RAC1P_2_1": rng.uniform(0, 0.2, n),
                "Sample Count": [50] * n,
            }
        )
        self.combined_df.index.name = "Partition ID"

    def test_returns_dataframe_and_list(self):
        plot_df, bar_colors = _prepare_value_plot_data(self.combined_df, ["SEX", "RAC1P"], value_colors=None)
        self.assertIsInstance(plot_df, pd.DataFrame)
        self.assertIsInstance(bar_colors, list)

    def test_plot_df_has_one_column_per_attribute(self):
        plot_df, _ = _prepare_value_plot_data(self.combined_df, ["SEX", "RAC1P"], value_colors=None)
        self.assertIn("SEX", plot_df.columns)
        self.assertIn("RAC1P", plot_df.columns)

    def test_plot_df_values_are_max_of_pairwise_cols(self):
        """Each row in plot_df should be the max across pairwise columns."""
        plot_df, _ = _prepare_value_plot_data(self.combined_df, ["SEX"], value_colors=None)
        expected = self.combined_df[["SEX_1_2", "SEX_2_1"]].max(axis=1)
        pd.testing.assert_series_equal(plot_df["SEX"], expected, check_names=False)

    def test_bar_colors_populated_when_value_colors_provided(self):
        """When value_colors is given, bar_colors should contain one list per attribute."""
        value_colors = {1: "red", 2: "blue"}
        _, bar_colors = _prepare_value_plot_data(self.combined_df, ["SEX", "RAC1P"], value_colors=value_colors)
        # One color list per attribute
        self.assertEqual(len(bar_colors), 2)
        # Each entry should be a list with n_partitions colors
        self.assertEqual(len(bar_colors[0]), len(self.combined_df))

    def test_bar_colors_empty_when_no_value_colors(self):
        """Without value_colors, bar_colors should be an empty list."""
        _, bar_colors = _prepare_value_plot_data(self.combined_df, ["SEX"], value_colors=None)
        self.assertEqual(bar_colors, [])

    def test_colors_are_valid_strings(self):
        """Each color should be a non-empty string."""
        value_colors = {1: "red", 2: "blue"}
        _, bar_colors = _prepare_value_plot_data(self.combined_df, ["SEX"], value_colors=value_colors)
        for color in bar_colors[0]:
            self.assertIsInstance(color, str)
            self.assertTrue(len(color) > 0)

    def test_unknown_group_defaults_to_gray(self):
        """If the bias direction doesn't match any key in value_colors, default to 'gray'."""
        # Both pairwise cols will have positive value, idxmax will be "SEX_1_2",
        # group_a will be "1" → int 1 → color "red" from value_colors
        # But with value_colors only containing key=99, it should default to gray
        value_colors = {99: "green"}
        _, bar_colors = _prepare_value_plot_data(self.combined_df, ["SEX"], value_colors=value_colors)
        # All colors should be gray since group key won't be in value_colors
        for color in bar_colors[0]:
            self.assertEqual(color, "gray")


class TestPlotMultiAttributeFairness(unittest.TestCase):
    """Tests for plot_multi_attribute_fairness."""

    def setUp(self):
        n = 30
        rng = np.random.default_rng(2)
        self.df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 1, n),
                "SEX": [1] * (n // 2) + [2] * (n // 2),
                "RAC1P": [1] * (n // 2) + [2] * (n // 2),
                "label": [0] * (n // 2) + [1] * (n // 2),
            }
        )
        self.partitioner = _make_mock_partitioner(self.df, num_partitions=3)

    def tearDown(self):
        plt.close("all")

    def test_returns_fig_ax_df(self):
        fig, ax, df = plot_multi_attribute_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            label_name="label",
            sens_atts=["SEX", "RAC1P"],
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsNotNone(ax)
        self.assertIsInstance(df, pd.DataFrame)

    def test_df_contains_fairness_columns(self):
        _, _, df = plot_multi_attribute_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            label_name="label",
            sens_atts=["SEX"],
        )
        # Should contain at least one column related to SEX
        self.assertTrue(any("SEX" in col for col in df.columns))

    def test_figure_is_not_empty(self):
        """Figure should have at least one axes object."""
        fig, _, _ = plot_multi_attribute_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            label_name="label",
            sens_atts=["SEX"],
        )
        self.assertGreater(len(fig.axes), 0)

    def test_respects_max_num_partitions(self):
        """With max_num_partitions=1, result df should have exactly 1 row."""
        _, _, df = plot_multi_attribute_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            label_name="label",
            sens_atts=["SEX"],
            max_num_partitions=1,
        )
        self.assertEqual(len(df), 1)

    def test_custom_title_is_set(self):
        """The axis title should reflect the custom title if provided."""
        _, ax, _ = plot_multi_attribute_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            label_name="label",
            sens_atts=["SEX"],
            title="My Custom Title",
        )
        self.assertEqual(ax.get_title(), "My Custom Title")

    def test_eo_metric(self):
        """Should work correctly with EO metric instead of the default DP."""
        fig, _, df = plot_multi_attribute_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            label_name="label",
            sens_atts=["SEX"],
            fairness_metric="EO",
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(any("EO" in col or "SEX" in col for col in df.columns))

    @patch("FeDa4Fair.metrics.fairness.compute_multi_fairness")
    def test_uses_compute_multi_fairness(self, mock_cmf):
        """plot_multi_attribute_fairness should call compute_multi_fairness internally."""
        # Return a minimal DataFrame
        mock_cmf.return_value = pd.DataFrame(
            {"SEX_DP": [0.1, 0.2], "SEX_val": ["1_2", "1_2"], "Sample Count": [50, 50]}
        )
        with contextlib.suppress(Exception):
            plot_multi_attribute_fairness(
                partitioner=self.partitioner,
                partitioner_test=self.partitioner,
                label_name="label",
                sens_atts=["SEX"],
            )
        mock_cmf.assert_called_once()


if __name__ == "__main__":
    unittest.main()
