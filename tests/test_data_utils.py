import unittest

import pandas as pd
import pytest

from FeDa4Fair.utils.data_utils import balance_data, generate_bias_by_groups, generate_modification_dict


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy data for balancing tests
        # sens=0: 75 True, 25 False (Rate 0.75)
        # sens=1: 25 True, 75 False (Rate 0.25)
        data = {"sens": [0] * 100 + [1] * 100, "label": [True] * 75 + [False] * 25 + [True] * 25 + [False] * 75}
        self.df = pd.DataFrame(data)

    def test_balance_data(self):
        # Min pos = 25, Min neg = 25
        # Both groups should end up with 25 pos, 25 neg.
        # Total size per group = 50. Rate = 25/50 = 0.5
        balanced_df, removed_count = balance_data(self.df, "sens", "label")

        # Check rates
        s0 = balanced_df[balanced_df["sens"] == 0]
        s1 = balanced_df[balanced_df["sens"] == 1]

        rate0 = s0["label"].sum() / len(s0)
        rate1 = s1["label"].sum() / len(s1)

        self.assertAlmostEqual(rate0, 0.5, delta=0.01)
        self.assertAlmostEqual(rate1, 0.5, delta=0.01)
        self.assertEqual(len(balanced_df), 100)  # 50 from each group

    def test_balance_data_with_integer_labels(self):
        """balance_data should work correctly when labels are integers (0/1),
        not just booleans."""
        data = {
            "sens": [0] * 100 + [1] * 100,
            "label": [1] * 75 + [0] * 25 + [1] * 25 + [0] * 75,  # integers, not booleans
        }
        df = pd.DataFrame(data)

        balanced_df, removed_count = balance_data(df, "sens", "label")

        # After balancing: min positive = 25, min negative = 25
        s0 = balanced_df[balanced_df["sens"] == 0]
        s1 = balanced_df[balanced_df["sens"] == 1]

        # Both groups should have equal positive counts
        self.assertEqual(s0["label"].sum(), s1["label"].sum())
        # Both groups should have equal sizes
        self.assertEqual(len(s0), len(s1))
        # Total should be 100 (25 pos + 25 neg per group)
        self.assertEqual(len(balanced_df), 100)
        # removed_count should be positive
        self.assertGreater(removed_count, 0)

    def test_drop_data_with_integer_labels(self):
        """drop_data should only drop rows where label == 1, 
        not all truthy values."""
        from FeDa4Fair.utils.data_utils import drop_data
        data = {
            "sens": [1] * 50 + [2] * 50,
            "label": [0, 1] * 50,  # integer labels
        }
        df = pd.DataFrame(data)
        original_len = len(df)

        result = drop_data(df, 0.5, "sens", 1, "label")

        # Should only affect rows where sens==1 AND label==1 (25 rows)
        # Drop 50% -> 12 or 13 dropped
        dropped = original_len - len(result)
        self.assertGreater(dropped, 0)
        self.assertLessEqual(dropped, 25)  # Can't drop more than matching rows
        
    def test_drop_data_with_integer_labels_count(self):
        from FeDa4Fair.utils.data_utils import drop_data
        data = {
            "sens": [1] * 50 + [2] * 50,
            "label": [0, 1] * 50,  # integer labels
        }
        df = pd.DataFrame(data)
        original_len = len(df)
        
        result = drop_data(df, 0.5, "sens", 1, "label")
        dropped = original_len - len(result)
        # Exact calculation: 25 matching rows * 0.5 = 12
        self.assertEqual(dropped, 12)

    def test_flip_data_with_integer_labels(self):
        """flip_data should correctly flip integer labels from 1 to 0."""
        from FeDa4Fair.utils.data_utils import flip_data
        data = {
            "sens": [1] * 50 + [2] * 50,
            "label": [0, 1] * 50,  # integer labels
        }
        df = pd.DataFrame(data)
        original_positive_count = df[df["label"] == 1].shape[0]

        result = flip_data(df.copy(), 0.5, "sens", 1, "label")

        # Some labels that were 1 should now be 0
        new_positive_count = result[result["label"] == 1].shape[0]
        self.assertLess(new_positive_count, original_positive_count)
        # Length should not change
        self.assertEqual(len(result), len(df))

    def test_generate_modification_dict_iid(self):
        num_clients = 5
        mod_dict = generate_modification_dict(
            client_ids=num_clients, attribute="sex", value=1, drop_rate_range=(0.1, 0.5)
        )

        self.assertEqual(len(mod_dict), 5)
        self.assertIn(0, mod_dict)
        self.assertIn(4, mod_dict)
        self.assertEqual(mod_dict[0]["sex"]["drop_rate"], 0.1)
        self.assertEqual(mod_dict[4]["sex"]["drop_rate"], 0.5)

    def test_generate_modification_dict_named(self):
        client_ids = ["A", "B", "C"]
        mod_dict = generate_modification_dict(
            client_ids=client_ids, attribute="sex", value=1, flip_rate_range=(0.2, 0.8)
        )

        self.assertEqual(len(mod_dict), 3)
        self.assertIn("A", mod_dict)
        self.assertEqual(mod_dict["A"]["sex"]["flip_rate"], 0.2)
        self.assertEqual(mod_dict["C"]["sex"]["flip_rate"], 0.8)

    def test_generate_bias_by_groups(self):
        group_configs = [
            {
                "group_id": "G1",
                "num_clients": 2,
                "sensitive_attr": "sex",
                "sensitive_value": 1,
                "drop_mean": 0.5,
                "drop_std": 0.1,
                "flip_mean": 0.2,
                "flip_std": 0.05,
            },
            {
                "group_id": "G2",
                "num_clients": 2,
                "sensitive_attr": "race",
                "sensitive_value": 2,
                "drop_mean": 0.1,
                "drop_std": 0.01,
                "flip_mean": 0.05,
                "flip_std": 0.01,
            },
        ]

        mod_dict = generate_bias_by_groups(num_total_clients=4, group_configs=group_configs)

        self.assertEqual(len(mod_dict), 4)
        # Client 0 and 1 should have 'sex' as key
        self.assertIn("sex", mod_dict[0])
        self.assertIn("sex", mod_dict[1])
        # Client 2 and 3 should have 'race' as key
        self.assertIn("race", mod_dict[2])
        self.assertIn("race", mod_dict[3])

        # Verify group_id is preserved
        self.assertEqual(mod_dict[0]["sex"]["group_id"], "G1")
        self.assertEqual(mod_dict[2]["race"]["group_id"], "G2")

    def test_generate_bias_by_groups_validation(self):
        group_configs = [{"num_clients": 1}]
        with pytest.raises(ValueError, match="Sum of group clients"):
            generate_bias_by_groups(num_total_clients=10, group_configs=group_configs)


if __name__ == "__main__":
    unittest.main()
