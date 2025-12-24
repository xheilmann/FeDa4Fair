import unittest
import pandas as pd
import numpy as np
from FeDa4Fair.utils.data_utils import balance_data, generate_bias_by_groups, generate_modification_dict

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy data for balancing tests
        # sens=0: 75 True, 25 False (Rate 0.75)
        # sens=1: 25 True, 75 False (Rate 0.25)
        data = {
            "sens": [0]*100 + [1]*100,
            "label": [True]*75 + [False]*25 + [True]*25 + [False]*75
        }
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
        self.assertEqual(len(balanced_df), 100) # 50 from each group

    def test_generate_modification_dict_iid(self):
        num_clients = 5
        mod_dict = generate_modification_dict(
            client_ids=num_clients,
            attribute="sex",
            value=1,
            drop_rate_range=(0.1, 0.5)
        )
        
        self.assertEqual(len(mod_dict), 5)
        self.assertIn(0, mod_dict)
        self.assertIn(4, mod_dict)
        self.assertEqual(mod_dict[0]["sex"]["drop_rate"], 0.1)
        self.assertEqual(mod_dict[4]["sex"]["drop_rate"], 0.5)

    def test_generate_modification_dict_named(self):
        client_ids = ["A", "B", "C"]
        mod_dict = generate_modification_dict(
            client_ids=client_ids,
            attribute="sex",
            value=1,
            flip_rate_range=(0.2, 0.8)
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
                "flip_std": 0.05
            },
            {
                "group_id": "G2",
                "num_clients": 2,
                "sensitive_attr": "race",
                "sensitive_value": 2,
                "drop_mean": 0.1,
                "drop_std": 0.01,
                "flip_mean": 0.05,
                "flip_std": 0.01
            }
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
        with self.assertRaises(ValueError):
            generate_bias_by_groups(num_total_clients=10, group_configs=group_configs)

if __name__ == "__main__":
    unittest.main()
