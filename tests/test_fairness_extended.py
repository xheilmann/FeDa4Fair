"""Tests for compute_multi_fairness, _evaluate_model_on_partition, and _evaluate_data_bias_on_partition."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from FeDa4Fair.metrics.fairness import (
    _evaluate_data_bias_on_partition,
    _evaluate_model_on_partition,
    compute_multi_fairness,
)


def _make_mock_partition(df: pd.DataFrame) -> MagicMock:
    """Helper: create a mock partition whose .to_pandas() returns df."""
    mock = MagicMock()
    mock.to_pandas.return_value = df
    mock.__len__ = MagicMock(return_value=len(df))
    return mock


def _make_mock_partitioner(df: pd.DataFrame, num_partitions: int = 2) -> MagicMock:
    """Helper: create a mock partitioner that always returns the same df."""
    partition = _make_mock_partition(df)
    partitioner = MagicMock()
    partitioner.load_partition.return_value = partition
    partitioner.num_partitions = num_partitions
    return partitioner


class TestEvaluateDataBiasOnPartition(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6],
                "sensitive": [0, 0, 0, 1, 1, 1],
                "label": [1, 0, 1, 0, 1, 0],
            }
        )
        self.partition = _make_mock_partition(self.df)

    def test_returns_pandas_series(self):
        result = _evaluate_data_bias_on_partition(self.partition, "sensitive", "DP", "label", "attribute")
        self.assertIsInstance(result, pd.Series)

    def test_output_contains_dp_key(self):
        result = _evaluate_data_bias_on_partition(self.partition, "sensitive", "DP", "label", "attribute")
        self.assertIn("sensitive_DP", result.index)

    def test_output_contains_eo_key(self):
        result = _evaluate_data_bias_on_partition(self.partition, "sensitive", "EO", "label", "attribute")
        self.assertIn("sensitive_EO", result.index)

    def test_dp_value_is_float(self):
        result = _evaluate_data_bias_on_partition(self.partition, "sensitive", "DP", "label", "attribute")
        self.assertIsInstance(float(result["sensitive_DP"]), float)

    def test_raises_on_non_dataframe_partition(self):
        """If partition.to_pandas() doesn't return a DataFrame, raise TypeError."""
        bad_partition = MagicMock()
        bad_partition.to_pandas.return_value = "not_a_dataframe"
        with pytest.raises(TypeError):
            _evaluate_data_bias_on_partition(bad_partition, "sensitive", "DP", "label", "attribute")

    def test_strips_index_artifact_column(self):
        """If __index_level_0__ is present in df, it should be silently dropped."""
        df_with_artifact = self.df.copy()
        df_with_artifact["__index_level_0__"] = range(len(df_with_artifact))
        partition = _make_mock_partition(df_with_artifact)
        # Should not raise
        result = _evaluate_data_bias_on_partition(partition, "sensitive", "DP", "label", "attribute")
        self.assertIn("sensitive_DP", result.index)


class TestEvaluateModelOnPartition(unittest.TestCase):
    def setUp(self):
        # Simple linearly separable data so any model can fit it
        rng = np.random.default_rng(0)
        n = 40
        self.df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 1, n),
                "f2": rng.uniform(0, 1, n),
                "sensitive": ([0] * (n // 2) + [1] * (n // 2)),
                "label": ([0] * (n // 2) + [1] * (n // 2)),
            }
        )
        self.train_partition = _make_mock_partition(self.df)
        self.test_partition = _make_mock_partition(self.df)

    def test_returns_pandas_series_with_accuracy(self):
        model = LogisticRegression(max_iter=200)
        result = _evaluate_model_on_partition(
            model,
            self.train_partition,
            self.test_partition,
            "sensitive",
            "DP",
            "label",
            sens_cols=["sensitive"],
            size_unit="attribute",
        )
        self.assertIsInstance(result, pd.Series)
        self.assertIn("Accuracy", result.index)
        self.assertIn("sensitive_DP", result.index)
        self.assertGreaterEqual(float(result["Accuracy"]), 0.0)
        self.assertLessEqual(float(result["Accuracy"]), 1.0)

    def test_raises_on_non_dataframe_partition(self):
        bad_partition = MagicMock()
        bad_partition.to_pandas.return_value = "not_a_dataframe"
        with pytest.raises(TypeError):
            _evaluate_model_on_partition(
                LogisticRegression(),
                bad_partition,
                self.test_partition,
                "sensitive",
                "DP",
                "label",
                sens_cols=["sensitive"],
                size_unit="attribute",
            )


class TestComputeMultiFairness(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 40
        self.df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 1, n),
                "SEX": ([1] * (n // 2) + [2] * (n // 2)),
                "RAC1P": ([1] * (n // 4) + [2] * (n // 4) + [1] * (n // 4) + [2] * (n // 4)),
                "label": ([0] * (n // 2) + [1] * (n // 2)),
            }
        )
        self.partitioner = _make_mock_partitioner(self.df, num_partitions=2)

    def test_returns_dataframe(self):
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=None,
            sens_atts=["SEX", "RAC1P"],
            label_name="label",
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_output_has_correct_number_of_rows(self):
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=None,
            sens_atts=["SEX"],
            label_name="label",
        )
        self.assertEqual(len(result), 2)  # 2 partitions

    def test_output_index_name_is_partition_id(self):
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=None,
            sens_atts=["SEX"],
            label_name="label",
        )
        self.assertEqual(result.index.name, "Partition ID")

    def test_output_contains_sample_count(self):
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=None,
            sens_atts=["SEX"],
            label_name="label",
        )
        self.assertIn("Sample Count", result.columns)

    def test_output_contains_fairness_columns_for_each_attr(self):
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=None,
            sens_atts=["SEX", "RAC1P"],
            label_name="label",
        )
        self.assertTrue(any("SEX" in col for col in result.columns))
        self.assertTrue(any("RAC1P" in col for col in result.columns))

    def test_with_model(self):
        model = LogisticRegression(max_iter=200)
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=model,
            sens_atts=["SEX"],
            label_name="label",
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Accuracy", result.columns)

    def test_max_num_partitions_limits_output(self):
        """max_num_partitions should cap the number of partitions evaluated."""
        result = compute_multi_fairness(
            partitioner=self.partitioner,
            partitioner_test=self.partitioner,
            model=None,
            sens_atts=["SEX"],
            label_name="label",
            max_num_partitions=1,
        )
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
