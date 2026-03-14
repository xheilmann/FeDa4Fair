"""Tests for cap_samples and generate_multiobjective_bias in data_utils.py."""

import unittest

import pandas as pd
import pytest

from FeDa4Fair.utils.data_utils import cap_samples, generate_multiobjective_bias


class TestCapSamples(unittest.TestCase):
    def setUp(self):
        # 200 rows: 100 positive (label=1), 100 negative (label=0)
        self.df = pd.DataFrame(
            {
                "feature": range(200),
                "label": [1] * 100 + [0] * 100,
            }
        )

    def test_returns_same_df_when_under_cap(self):
        """If the DataFrame is smaller than the cap, return it unchanged."""
        result = cap_samples(self.df, cap=300, label_column="label")
        self.assertEqual(len(result), 200)

    def test_returns_same_df_when_at_cap(self):
        """If the DataFrame is exactly the cap, return it unchanged."""
        result = cap_samples(self.df, cap=200, label_column="label")
        self.assertEqual(len(result), 200)

    def test_reduces_to_cap_size(self):
        """Result should contain exactly `cap` rows."""
        result = cap_samples(self.df, cap=100, label_column="label")
        self.assertEqual(len(result), 100)

    def test_preserves_label_distribution(self):
        """The label proportions should be approximately maintained after capping."""
        # Original: 50% label=1, 50% label=0
        result = cap_samples(self.df, cap=100, label_column="label")
        pos_rate = result["label"].sum() / len(result)
        # Should be close to 0.5 (within 5%)
        self.assertAlmostEqual(pos_rate, 0.5, delta=0.05)

    def test_preserves_label_distribution_imbalanced(self):
        """Distribution should be maintained even when labels are imbalanced."""
        # 150 positive, 50 negative (75% / 25%)
        df_imbalanced = pd.DataFrame(
            {
                "feature": range(200),
                "label": [1] * 150 + [0] * 50,
            }
        )
        result = cap_samples(df_imbalanced, cap=80, label_column="label")
        self.assertEqual(len(result), 80)
        pos_rate = result["label"].sum() / len(result)
        # Should be close to 0.75
        self.assertAlmostEqual(pos_rate, 0.75, delta=0.1)

    def test_is_deterministic_with_same_seed(self):
        """Same seed should produce the same result."""
        r1 = cap_samples(self.df, cap=50, label_column="label", seed=42)
        r2 = cap_samples(self.df, cap=50, label_column="label", seed=42)
        self.assertTrue(r1.equals(r2))

    def test_different_seeds_produce_different_results(self):
        """Different seeds should generally produce different subsets."""
        r1 = cap_samples(self.df, cap=50, label_column="label", seed=0)
        r2 = cap_samples(self.df, cap=50, label_column="label", seed=99)
        # It is astronomically unlikely for two random subsets to be identical
        self.assertFalse(r1.index.tolist() == r2.index.tolist())

    def test_does_not_mutate_input(self):
        """cap_samples should not modify the original DataFrame."""
        original_len = len(self.df)
        original_index = self.df.index.tolist().copy()
        cap_samples(self.df, cap=50, label_column="label")
        self.assertEqual(len(self.df), original_len)
        self.assertEqual(self.df.index.tolist(), original_index)

    def test_returns_a_copy(self):
        """The returned DataFrame should be a copy, not a view."""
        result = cap_samples(self.df, cap=50, label_column="label")
        # Modifying result should not affect self.df
        result.loc[result.index[0], "feature"] = -999
        self.assertNotIn(-999, self.df["feature"].values)

    def test_boolean_label_column(self):
        """Should work with boolean label columns."""
        df_bool = pd.DataFrame(
            {
                "feature": range(100),
                "label": [True] * 60 + [False] * 40,
            }
        )
        result = cap_samples(df_bool, cap=50, label_column="label")
        self.assertEqual(len(result), 50)


class TestGenerateMultiobjectiveBias(unittest.TestCase):
    def test_basic_two_groups_two_attrs(self):
        """Should generate a mod_dict with the correct structure for two groups."""
        group_configs = [
            {
                "group_id": "biased",
                "num_clients": 3,
                "configs": [
                    {
                        "attribute": "SEX",
                        "value": 2,
                        "drop_mean": 0.4,
                        "drop_std": 0.05,
                        "flip_mean": 0.1,
                        "flip_std": 0.01,
                    },
                    {
                        "attribute": "RAC1P",
                        "value": 3,
                        "drop_mean": 0.2,
                        "drop_std": 0.02,
                        "flip_mean": 0.0,
                        "flip_std": 0.0,
                    },
                ],
            },
            {
                "group_id": "mitigated",
                "num_clients": 2,
                "configs": [
                    {"attribute": "SEX", "mitigate": True},
                ],
            },
        ]

        mod_dict = generate_multiobjective_bias(num_total_clients=5, group_configs=group_configs)

        self.assertEqual(len(mod_dict), 5)

        # First 3 clients should have both SEX and RAC1P
        for client_id in range(3):
            self.assertIn("SEX", mod_dict[client_id])
            self.assertIn("RAC1P", mod_dict[client_id])
            # Values should be floats in [0, 1]
            self.assertGreaterEqual(mod_dict[client_id]["SEX"]["drop_rate"], 0.0)
            self.assertLessEqual(mod_dict[client_id]["SEX"]["drop_rate"], 1.0)

        # Last 2 clients should have SEX with mitigate=True
        for client_id in [3, 4]:
            self.assertIn("SEX", mod_dict[client_id])
            self.assertTrue(mod_dict[client_id]["SEX"]["mitigate"])

    def test_raises_on_client_count_mismatch(self):
        """Raises ValueError if group client counts don't sum to total."""
        group_configs = [
            {"group_id": "g1", "num_clients": 2, "configs": []},
        ]
        with pytest.raises(ValueError, match="Sum of group clients"):
            generate_multiobjective_bias(num_total_clients=5, group_configs=group_configs)

    def test_zero_std_uses_mean_exactly(self):
        """When std=0, all clients in the group should get exactly the mean."""
        group_configs = [
            {
                "group_id": "g1",
                "num_clients": 3,
                "configs": [
                    {
                        "attribute": "SEX",
                        "value": 2,
                        "drop_mean": 0.5,
                        "drop_std": 0.0,
                        "flip_mean": 0.2,
                        "flip_std": 0.0,
                    }
                ],
            }
        ]
        mod_dict = generate_multiobjective_bias(num_total_clients=3, group_configs=group_configs)
        for client_id in range(3):
            self.assertAlmostEqual(mod_dict[client_id]["SEX"]["drop_rate"], 0.5)
            self.assertAlmostEqual(mod_dict[client_id]["SEX"]["flip_rate"], 0.2)

    def test_with_custom_client_names(self):
        """Should support custom client name lists."""
        group_configs = [
            {
                "group_id": "g1",
                "num_clients": 2,
                "configs": [{"attribute": "SEX", "value": 1, "drop_mean": 0.3, "drop_std": 0.0, "flip_mean": 0.0, "flip_std": 0.0}],
            }
        ]
        mod_dict = generate_multiobjective_bias(
            num_total_clients=2, group_configs=group_configs, client_names=["alice", "bob"]
        )
        self.assertIn("alice", mod_dict)
        self.assertIn("bob", mod_dict)

    def test_group_id_preserved_in_output(self):
        """The group_id should be stored in each modification entry."""
        group_configs = [
            {
                "group_id": "MY_GROUP",
                "num_clients": 2,
                "configs": [
                    {"attribute": "SEX", "value": 2, "drop_mean": 0.1, "drop_std": 0.0, "flip_mean": 0.0, "flip_std": 0.0}
                ],
            }
        ]
        mod_dict = generate_multiobjective_bias(num_total_clients=2, group_configs=group_configs)
        self.assertEqual(mod_dict[0]["SEX"]["group_id"], "MY_GROUP")
        self.assertEqual(mod_dict[1]["SEX"]["group_id"], "MY_GROUP")

    def test_empty_configs_no_attrs(self):
        """Clients with empty configs list should have an empty mod entry."""
        group_configs = [
            {"group_id": "empty", "num_clients": 2, "configs": []},
        ]
        mod_dict = generate_multiobjective_bias(num_total_clients=2, group_configs=group_configs)
        self.assertEqual(mod_dict[0], {})
        self.assertEqual(mod_dict[1], {})


if __name__ == "__main__":
    unittest.main()
