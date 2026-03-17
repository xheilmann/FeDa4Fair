# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset generation and preprocessing for cross-silo and cross-device federated learning."""

from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset
from FeDa4Fair.metrics.evaluation import evaluate_models_on_datasets
from FeDa4Fair.utils.constants import (
    ACS_INCOME,
    ACS_RACE,
    CROSS_SILO,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEMOGRAPHIC_PARITY,
    INCOME_LABEL,
    LEVEL_ATTRIBUTE,
    MARITAL_STATUS,
    MIN_BIAS_THRESHOLD,
    RACE,
    RACE_BIAS_LOWER_BOUND,
    RACE_BIAS_UPPER_BOUND,
    SEX,
)
from FeDa4Fair.utils.data_types import ProcessedSiloData

# Standard mapping for ACS data preprocessing
ACS_MAPPING = {MARITAL_STATUS: {3: 2, 4: 2, 5: 2}, ACS_RACE: {8: 5, 7: 5, 9: 5, 6: 3, 5: 4, 3: 4}}


def split_df(df: pd.DataFrame, split_number: int) -> list[pd.DataFrame]:
    """
    Split a DataFrame into a specified number of approximately equal parts.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    split_number : int
        The number of splits to create.

    Returns
    -------
    list[pd.DataFrame]
        A list of DataFrames.

    """
    if split_number <= 0:
        return [df]

    # Calculate chunk size
    total_samples = len(df)
    chunk_size = total_samples // split_number
    remainder = total_samples % split_number

    splits = []
    start = 0
    for i in range(split_number):
        # Distribute remainder across the first few chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        splits.append(df.iloc[start:end].copy())
        start = end

    return splits


def create_cross_silo_data(
    fairness_level: Literal["attribute", "value", "attribute-value"], path: PathLike | str
) -> None:
    """
    Generate, evaluate, and save cross-silo datasets with varying bias levels.
    """
    base_path = Path(path)
    datasets = _get_initial_datasets(fairness_level, base_path)

    stats_df, _fig = evaluate_models_on_datasets(datasets, n_jobs=3, fairness_level=fairness_level)
    stats_df.to_csv(base_path / "data_stats" / f"crosssilo_{fairness_level}_0.0.csv", index=False)

    all_modifications = []
    for drop_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        prev_dr = np.round(drop_rate - 0.1, 2)
        prev_stats_file = base_path / "data_stats" / f"crosssilo_{fairness_level}_{prev_dr}.csv"
        current_stats_df = pd.read_csv(prev_stats_file)

        states, _partitioners, modification_dict, silo_modifications = _determine_modifications(
            current_stats_df, drop_rate, fairness_level
        )
        all_modifications.extend(silo_modifications)

        if len(states) <= 1:
            break

        datasets = _apply_silo_modifications(states, modification_dict, fairness_level, base_path)
        stats_df, _fig = evaluate_models_on_datasets(datasets, n_jobs=3, fairness_level=fairness_level)
        stats_df.to_csv(base_path / "data_stats" / f"crosssilo_{fairness_level}_{drop_rate}.csv", index=False)

    all_modifications_df = pd.DataFrame(all_modifications, columns=["state", "drop_rate", "attribute", "value"])
    all_modifications_df.to_csv(base_path / "data_stats" / f"crosssilo_{fairness_level}_modifications.csv", index=False)


def _get_initial_datasets(fairness_level: str, base_path: Path) -> list[ProcessedSiloData]:
    fair_dataset = FairFederatedDataset(
        dataset=ACS_INCOME,
        fl_setting=None,
        partitioners=None,
        fairness_metric=DEMOGRAPHIC_PARITY,
        fairness_level=fairness_level,
        mapping=ACS_MAPPING,
        path=base_path / "data" / f"cross_silo_{fairness_level}_final",
    )
    datasets = []
    if fair_dataset._states is None:
        msg = "States must be defined for ACS dataset."
        raise ValueError(msg)

    for state in fair_dataset._states:
        split_name = f"{state}_train" if fair_dataset._fl_setting == CROSS_SILO else state
        data = fair_dataset.load_partition(0, split_name).to_pandas()
        if isinstance(data, pd.DataFrame):
            datasets = preprocess_data_cross_silo(data, datasets, fairness_level, state)
    return datasets


def _determine_modifications(stats_df: pd.DataFrame, drop_rate: float, fairness_level: str):
    states, partitioners, modification_dict, silo_modifications = [], {}, {}, []
    for entry in stats_df["dataset"].unique():
        df_entry = stats_df[(stats_df["dataset"] == entry) & stats_df["model"].isin(["XGBoost", "LogisticRegression"])]

        if fairness_level == LEVEL_ATTRIBUTE:
            res = _get_attribute_modifications(entry, df_entry, drop_rate)
        else:
            res = _get_value_modifications(entry, df_entry, drop_rate)

        if res:
            state, mod_config, silo_mod = res
            states.append(state)
            partitioners[state] = 1
            modification_dict[state] = mod_config
            silo_modifications.append(silo_mod)
    return states, partitioners, modification_dict, silo_modifications


def _get_attribute_modifications(entry: str, df_entry: pd.DataFrame, drop_rate: float):
    race_dp = df_entry[f"{DEMOGRAPHIC_PARITY}_{RACE}"].values
    sex_dp = df_entry[f"{DEMOGRAPHIC_PARITY}_{SEX}"].values
    count = sum(1 for i in range(len(df_entry)) if sex_dp[i] > race_dp[i])

    # If sex bias is dominant across models
    attr = SEX if count > 0 else ACS_RACE
    target_value = 2  # Default discriminated group value

    modification_config = {
        attr: {"drop_rate": drop_rate, "flip_rate": 0, "value": target_value, "attribute": None, "attribute_value": None}
    }
    return entry, modification_config, [entry, drop_rate, attr, target_value]


def _get_value_modifications(entry: str, df_entry: pd.DataFrame, drop_rate: float):
    # value_DP_RACE has the format "first_last" (e.g. "1_2").
    val_col = f"value_{DEMOGRAPHIC_PARITY}_{RACE}"
    raw_first = str(df_entry[val_col].values[0])
    raw_second = str(df_entry[val_col].values[1])

    parts_first = raw_first.split("_")
    parts_second = raw_second.split("_")

    # The "discriminated" value is always the last token.
    v1, v2 = parts_first[-1], parts_second[-1]

    if v1 == v2:
        min_bias = np.min(df_entry[f"{DEMOGRAPHIC_PARITY}_{RACE}"].values)
        if min_bias < MIN_BIAS_THRESHOLD:
            # Use the first token of the first entry as the target value
            target_val = int(parts_first[0])
            mod_config = {
                ACS_RACE: {
                    "drop_rate": drop_rate,
                    "flip_rate": 0,
                    "value": target_val,
                    "attribute": None,
                    "attribute_value": None,
                }
            }
            return entry, mod_config, [entry, drop_rate, ACS_RACE, target_val]
    else:
        target_val = int(v1)
        mod_config = {
            ACS_RACE: {
                "drop_rate": drop_rate,
                "flip_rate": 0,
                "value": target_val,
                "attribute": None,
                "attribute_value": None,
            }
        }
        return entry, mod_config, [entry, drop_rate, ACS_RACE, target_val]
    return None


def _apply_silo_modifications(states: list[str], modification_dict: dict, fairness_level: str, base_path: Path):
    fair_dataset = FairFederatedDataset(
        dataset=ACS_INCOME,
        fl_setting=CROSS_SILO,
        partitioners=None,
        states=states,
        fairness_metric=DEMOGRAPHIC_PARITY,
        fairness_level=fairness_level,
        modification_dict=modification_dict,
        mapping=ACS_MAPPING,
        path=base_path / "data" / f"cross_silo_{fairness_level}_final",
    )
    datasets = []
    if fair_dataset._states is None:
        msg = "States must be defined for ACS dataset."
        raise ValueError(msg)
    for state in fair_dataset._states:
        split_name = f"{state}_train" if fair_dataset._fl_setting == CROSS_SILO else state
        data = fair_dataset.load_partition(0, split_name).to_pandas()
        if isinstance(data, pd.DataFrame):
            datasets = preprocess_data_cross_silo(data, datasets, fairness_level, state)
    return datasets


def preprocess_data_cross_silo(
    data: pd.DataFrame, datasets: list[ProcessedSiloData], fairness_level: str, state: str
) -> list[ProcessedSiloData]:
    """
    Preprocess data for cross-silo federated learning.
    """
    target = data[INCOME_LABEL]
    features = data.drop(columns=[INCOME_LABEL])

    stratify_cols = [SEX, ACS_RACE] if fairness_level == LEVEL_ATTRIBUTE else None

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED, stratify=features[stratify_cols] if stratify_cols else None
    )

    sf_data = {SEX: X_test[SEX].values, RACE: X_test[ACS_RACE].values, MARITAL_STATUS: X_test[MARITAL_STATUS].values}

    # Drop sensitive features from training data
    X_train_clean = X_train.drop(columns=[MARITAL_STATUS, SEX, ACS_RACE])
    X_test_clean = X_test.drop(columns=[MARITAL_STATUS, SEX, ACS_RACE])

    datasets.append(
        ProcessedSiloData(
            name=state,
            x_train=X_train_clean.values,
            y_train=y_train.values,
            x_test=X_test_clean.values,
            y_test=y_test.values,
            sf_data=sf_data,
        )
    )
    return datasets


def create_cross_device_data(
    fairness_level: Literal["attribute", "value", "attribute-value"], split_number: int, path: PathLike | str
) -> None:
    """
    Generate, evaluate, and save cross-device datasets based on previously generated cross-silo data.
    """
    base_path = Path(path)
    datasets_all = _load_silo_datasets(fairness_level, split_number, base_path)

    stats_df, _fig = evaluate_models_on_datasets(datasets_all, n_jobs=3, fairness_level=fairness_level)
    stats_df.to_csv(base_path / "data_stats" / f"crossdevice_{fairness_level}.csv", index=False)

    states = _filter_states_by_fairness(stats_df, fairness_level)

    final_output_path = base_path / "data" / f"cross_device_{fairness_level}_final"
    final_output_path.mkdir(parents=True, exist_ok=True)

    input_data_path = base_path / "data" / f"cross-device-{fairness_level}"

    for state in states:
        state_file = input_data_path / f"{state}.csv"
        if state_file.exists():
            data = pd.read_csv(state_file)
            data.to_csv(final_output_path / f"{state}.csv", index=False)

    filtered_stats_df = stats_df[stats_df["dataset"].isin(states)]
    filtered_stats_df.to_csv(final_output_path / "model_perf_DP.csv", index=False)


def _load_silo_datasets(fairness_level: str, split_number: int, base_path: Path) -> list[ProcessedSiloData]:
    datasets_all = []
    silo_data_dir = base_path / "data" / f"cross_silo_{fairness_level}_final"

    if not silo_data_dir.exists():
        return []

    excluded_files = ["model_perf_DP.csv", "crosssilo_value_modifications.csv", "unfairness_distribution_DP.csv"]

    for file_path in silo_data_dir.iterdir():
        if file_path.name in excluded_files or not file_path.name.endswith(".csv"):
            continue

        data = pd.read_csv(file_path)
        datasets = preprocess_datasets(file_path.name, data, base_path, split_number, fairness_level)
        datasets_all.extend(datasets)
    return datasets_all


def _filter_states_by_fairness(stats_df: pd.DataFrame, fairness_level: str) -> list[str]:
    if fairness_level == LEVEL_ATTRIBUTE:
        return _filter_attribute_fairness(stats_df)
    return _filter_value_fairness(stats_df)


def _filter_attribute_fairness(stats_df: pd.DataFrame) -> list[str]:
    eligible_states = []
    for entry in stats_df["dataset"].unique():
        df_entry = stats_df[(stats_df["dataset"] == entry) & stats_df["model"].isin(["XGBoost", "LogisticRegression"])]
        race_dp = df_entry[f"{DEMOGRAPHIC_PARITY}_{RACE}"].values
        sex_dp = df_entry[f"{DEMOGRAPHIC_PARITY}_{SEX}"].values
        count_sex_biased = sum(1 for i in range(len(df_entry)) if sex_dp[i] > race_dp[i])

        if count_sex_biased == 2:
            if np.min(sex_dp) > MIN_BIAS_THRESHOLD:
                eligible_states.append(entry)
        elif count_sex_biased == 0 and RACE_BIAS_UPPER_BOUND > np.min(race_dp) > RACE_BIAS_LOWER_BOUND:
            eligible_states.append(entry)
    return eligible_states


def _filter_value_fairness(stats_df: pd.DataFrame) -> list[str]:
    eligible_states = []
    val_col = f"value_{DEMOGRAPHIC_PARITY}_{RACE}"
    for entry in stats_df["dataset"].unique():
        df_entry = stats_df[(stats_df["dataset"] == entry) & stats_df["model"].isin(["XGBoost", "LogisticRegression"])]
        if (
            df_entry[val_col].values[0][-3:-2] == df_entry[val_col].values[1][-3:-2]
            and np.min(df_entry[f"{DEMOGRAPHIC_PARITY}_{RACE}"].values) > MIN_BIAS_THRESHOLD
        ):
            eligible_states.append(entry)
    return eligible_states


def preprocess_datasets(
    filename: str, data: pd.DataFrame, base_path: Path, split_number: int = 6, fairness_level: str = "attribute"
) -> list[ProcessedSiloData]:
    """
    Split a dataset into multiple parts and preprocess each for cross-device evaluation.
    """
    split_datasets = split_df(data, split_number)
    processed_datasets = []
    device_data_path = base_path / "data" / f"cross-device-{fairness_level}"
    device_data_path.mkdir(parents=True, exist_ok=True)

    prefix = filename[:2]

    for i, split_data in enumerate(split_datasets):
        device_id = f"{prefix}_{i}"
        split_data.to_csv(device_data_path / f"{device_id}.csv", index=False)

        target = split_data[INCOME_LABEL]
        features = split_data.drop(columns=[INCOME_LABEL])

        stratify_cols = [SEX, ACS_RACE] if fairness_level == LEVEL_ATTRIBUTE else None

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED,
            stratify=features[stratify_cols] if stratify_cols else None
        )

        sf_data = {SEX: X_test[SEX].values, RACE: X_test[ACS_RACE].values, MARITAL_STATUS: X_test[MARITAL_STATUS].values}

        # Drop sensitive features from training data
        X_train_clean = X_train.drop(columns=[MARITAL_STATUS, SEX, ACS_RACE])
        X_test_clean = X_test.drop(columns=[MARITAL_STATUS, SEX, ACS_RACE])

        processed_datasets.append(
            ProcessedSiloData(
                name=device_id,
                x_train=X_train_clean.values,
                y_train=y_train.values,
                x_test=X_test_clean.values,
                y_test=y_test.values,
                sf_data=sf_data,
            )
        )
    return processed_datasets
