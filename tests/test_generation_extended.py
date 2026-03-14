"""Tests for _get_attribute_modifications and _get_value_modifications in generation.py."""

import unittest

import pandas as pd

from FeDa4Fair.dataset.generation import _get_attribute_modifications, _get_value_modifications


def _make_df_entry(dp_sex: list[float], dp_race: list[float], value_dp_race: list[str] | None = None) -> pd.DataFrame:
    """Helper: build a df_entry DataFrame matching the shape expected by these functions."""
    data = {
        "DP_SEX": dp_sex,
        "DP_RACE": dp_race,
        "model": ["XGBoost", "LogisticRegression"][: len(dp_sex)],
    }
    if value_dp_race is not None:
        data["value_DP_RACE"] = value_dp_race
    return pd.DataFrame(data)


class TestGetAttributeModifications(unittest.TestCase):
    DR = 0.3  # Example drop rate

    def test_both_sex_dp_dominate_returns_sex_attr(self):
        """When sex_dp > race_dp for all models (count==2), should target SEX."""
        # DP_SEX > DP_RACE for both rows → count = 2
        df_entry = _make_df_entry(dp_sex=[0.2, 0.25], dp_race=[0.05, 0.06])
        result = _get_attribute_modifications("StateA", df_entry, self.DR)
        self.assertIsNotNone(result)
        entry, mod, _ = result
        self.assertEqual(entry, "StateA")
        self.assertIn("SEX", mod)
        self.assertEqual(mod["SEX"]["drop_rate"], self.DR)

    def test_returns_none_when_sex_dp_exceeds_threshold(self):
        """When count==2 and min(sex_dp) >= 0.09, return None (already fair enough)."""
        df_entry = _make_df_entry(dp_sex=[0.15, 0.20], dp_race=[0.05, 0.06])
        result = _get_attribute_modifications("StateB", df_entry, self.DR)
        self.assertIsNone(result)

    def test_race_dp_dominates_returns_rac1p_attr(self):
        """When race_dp > sex_dp for all models (count==0), should target RAC1P."""
        # DP_RACE > DP_SEX for both rows → count = 0
        df_entry = _make_df_entry(dp_sex=[0.02, 0.03], dp_race=[0.06, 0.07])
        result = _get_attribute_modifications("StateC", df_entry, self.DR)
        self.assertIsNotNone(result)
        _, mod, _ = result
        self.assertIn("RAC1P", mod)

    def test_returns_none_when_race_dp_exceeds_threshold(self):
        """When count==0 and min(race_dp) >= 0.09, return None."""
        df_entry = _make_df_entry(dp_sex=[0.01, 0.02], dp_race=[0.12, 0.15])
        result = _get_attribute_modifications("StateD", df_entry, self.DR)
        self.assertIsNone(result)

    def test_mixed_count_1_uses_sex(self):
        """When count==1 (one model has sex_dp > race_dp), should target SEX."""
        # Only first row has sex_dp > race_dp
        df_entry = _make_df_entry(dp_sex=[0.05, 0.01], dp_race=[0.02, 0.04])
        result = _get_attribute_modifications("StateE", df_entry, self.DR)
        self.assertIsNotNone(result)
        _, mod, _ = result
        self.assertIn("SEX", mod)

    def test_silo_mod_has_correct_format(self):
        """The silo_mod list should have 4 elements: [entry, dr, attr, value]."""
        df_entry = _make_df_entry(dp_sex=[0.05, 0.06], dp_race=[0.01, 0.02])
        result = _get_attribute_modifications("StateF", df_entry, self.DR)
        self.assertIsNotNone(result)
        _, _, silo_mod = result
        self.assertEqual(len(silo_mod), 4)
        self.assertEqual(silo_mod[0], "StateF")
        self.assertEqual(silo_mod[1], self.DR)


class TestGetValueModifications(unittest.TestCase):
    DR = 0.3

    def test_same_last_token_low_min_dp_uses_first_token(self):
        """When v1==v2 and min_dp < 0.09, val should be the first token of raw0."""
        # value_DP_RACE = ["1_2", "3_2"] → v1="2", v2="2" → same
        # min(DP_RACE) = 0.05 < 0.09 → should return with val=1 (first token of "1_2")
        df_entry = _make_df_entry(
            dp_sex=[0.01, 0.01],
            dp_race=[0.05, 0.07],
            value_dp_race=["1_2", "3_2"],
        )
        result = _get_value_modifications("StateG", df_entry, self.DR)
        self.assertIsNotNone(result)
        entry, mod, _ = result
        self.assertEqual(entry, "StateG")
        self.assertIn("RAC1P", mod)
        # val = int("1") = 1
        self.assertEqual(mod["RAC1P"]["value"], 1)

    def test_same_last_token_high_min_dp_returns_none(self):
        """When v1==v2 but min_dp >= 0.09, should return None."""
        df_entry = _make_df_entry(
            dp_sex=[0.01, 0.01],
            dp_race=[0.12, 0.14],
            value_dp_race=["1_2", "3_2"],
        )
        result = _get_value_modifications("StateH", df_entry, self.DR)
        self.assertIsNone(result)

    def test_different_last_tokens_uses_first_entry_last_token(self):
        """When v1 != v2, val = last token of raw0 (the most unfair pair)."""
        # value_DP_RACE = ["1_2", "1_3"] → v1="2", v2="3" → different
        df_entry = _make_df_entry(
            dp_sex=[0.01, 0.01],
            dp_race=[0.05, 0.07],
            value_dp_race=["1_2", "1_3"],
        )
        result = _get_value_modifications("StateI", df_entry, self.DR)
        self.assertIsNotNone(result)
        _, mod, _ = result
        self.assertIn("RAC1P", mod)
        self.assertEqual(mod["RAC1P"]["value"], 2)

    def test_works_with_multi_digit_values(self):
        """Should handle multi-digit race values like '10_20' correctly (regression for BUG-2)."""
        # Old code: "10_20"[-3:-2] → "1" (wrong). New code: "10_20".split("_")[-1] → "20" (correct)
        df_entry = _make_df_entry(
            dp_sex=[0.01, 0.01],
            dp_race=[0.05, 0.07],
            value_dp_race=["10_20", "10_30"],
        )
        result = _get_value_modifications("StateJ", df_entry, self.DR)
        self.assertIsNotNone(result)
        _, mod, _ = result
        # v1="20", v2="30" → different → val = int("20") = 20
        self.assertEqual(mod["RAC1P"]["value"], 20)

    def test_returns_correct_silo_mod_format(self):
        """silo_mod should be [entry, dr, "RAC1P", val]."""
        df_entry = _make_df_entry(
            dp_sex=[0.01, 0.01],
            dp_race=[0.05, 0.07],
            value_dp_race=["1_2", "1_3"],
        )
        result = _get_value_modifications("StateK", df_entry, self.DR)
        self.assertIsNotNone(result)
        _, _, silo_mod = result
        self.assertEqual(silo_mod[0], "StateK")
        self.assertEqual(silo_mod[1], self.DR)
        self.assertEqual(silo_mod[2], "RAC1P")


if __name__ == "__main__":
    unittest.main()
