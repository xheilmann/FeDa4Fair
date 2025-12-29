from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset
from FeDa4Fair.metrics.evaluation import evaluate_models_on_datasets

mapping = {"MAR": {3: 2, 4: 2, 5: 2}, "RAC1P": {8: 5, 7: 5, 9: 5, 6: 3, 5: 4, 3: 4}}


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
    n = len(df)
    chunk_size = n // split_number
    remainder = n % split_number

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
    path_str = str(path)
    datasets = _get_initial_datasets(fairness_level, path_str)

    df, fig = evaluate_models_on_datasets(datasets, n_jobs=3, fairness_level=fairness_level)
    df.to_csv(f"{path_str}data_stats/crosssilo_{fairness_level}_0.0.csv", index=False)

    all_modifications = []
    for dr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        df = pd.read_csv(f"{path_str}data_stats/crosssilo_{fairness_level}_{np.round(dr - 0.1, 2)}.csv")
        states, partitioners, modification_dict, silo_modifications = _determine_modifications(df, dr, fairness_level)
        all_modifications.extend(silo_modifications)

        if len(states) <= 1:
            break

        datasets = _apply_silo_modifications(states, modification_dict, fairness_level, path_str)
        df, fig = evaluate_models_on_datasets(datasets, n_jobs=3, fairness_level=fairness_level)
        df.to_csv(f"{path_str}data_stats/crosssilo_{fairness_level}_{dr}.csv", index=False)

    all_modifications_df = pd.DataFrame(all_modifications, columns=["state", "drop_rate", "attribute", "value"])  # type: ignore[arg-type]
    all_modifications_df.to_csv(Path(f"{path_str}data_stats/crosssilo_{fairness_level}_modifications.csv"), index=False)


def _get_initial_datasets(fairness_level, path_str):
    ffds = FairFederatedDataset(
        dataset="ACSIncome",
        fl_setting=None,
        partitioners={"train": 1},
        fairness_metric="DP",
        fairness_level=fairness_level,
        mapping=mapping,
        path=Path(f"{path_str}data/cross_silo_{fairness_level}_final"),
    )
    datasets = []
    if ffds._states is None:
        msg = "States must be defined for ACS dataset."
        raise ValueError(msg)
    for state in ffds._states:
        data = ffds.load_partition(0, state).to_pandas()
        if isinstance(data, pd.DataFrame):
            datasets = preprocess_data_cross_silo(data, datasets, fairness_level, state)
    return datasets


def _determine_modifications(df, dr, fairness_level):
    states, partitioners, modification_dict, silo_modifications = [], {}, {}, []
    for entry in df["dataset"].unique():
        df_entry = df[(df["dataset"] == entry) & df["model"].isin(["XGBoost", "LogisticRegression"])]

        if fairness_level == "attribute":
            res = _get_attribute_modifications(entry, df_entry, dr)
        else:
            res = _get_value_modifications(entry, df_entry, dr)

        if res:
            state, mod, silo_mod = res
            states.append(state)
            partitioners[state] = 1
            modification_dict[state] = mod
            silo_modifications.append(silo_mod)
    return states, partitioners, modification_dict, silo_modifications


def _get_attribute_modifications(entry, df_entry, dr):
    race_dp = df_entry["DP_RACE"].values
    sex_dp = df_entry["DP_SEX"].values
    count = sum(1 for i in range(len(df_entry)) if sex_dp[i] > race_dp[i])

    if count == 2 or count != 0:  # Combined count == 2 and 'else' case from original
        min_val = np.min(sex_dp)
        if count == 2 and min_val >= 0.09:
            return None
        attr = "SEX"
    else:  # count == 0
        min_val = np.min(race_dp)
        if min_val >= 0.09:
            return None
        attr = "RAC1P"

    mod = {attr: {"drop_rate": dr, "flip_rate": 0, "value": 2, "attribute": None, "attribute_value": None}}
    return entry, mod, [entry, dr, attr, 2]


def _get_value_modifications(entry, df_entry, dr):
    v1, v2 = df_entry["value_DP_RACE"].values[0][-3:-2], df_entry["value_DP_RACE"].values[1][-3:-2]
    if v1 == v2:
        min_val = np.min(df_entry["DP_RACE"].values)
        if min_val < 0.09:
            val = int(df_entry["value_DP_RACE"].values[0][:1])
            mod = {"RAC1P": {"drop_rate": dr, "flip_rate": 0, "value": val, "attribute": None, "attribute_value": None}}
            return entry, mod, [entry, dr, "RAC1P", val]
    else:
        val = int(df_entry["value_DP_RACE"].values[0][-3:-2])
        mod = {"RAC1P": {"drop_rate": dr, "flip_rate": 0, "value": val, "attribute": None, "attribute_value": None}}
        return entry, mod, [entry, dr, "RAC1P", val]
    return None


def _apply_silo_modifications(states, modification_dict, fairness_level, path_str):
    ffds = FairFederatedDataset(
        dataset="ACSIncome",
        fl_setting="cross-silo",
        partitioners={"train": 1},
        states=states,
        fairness_metric="DP",
        fairness_level=fairness_level,
        modification_dict=modification_dict,
        mapping=mapping,
        path=Path(f"{path_str}data/cross_silo_{fairness_level}_final"),
    )
    datasets = []
    if ffds._states is None:
        msg = "States must be defined for ACS dataset."
        raise ValueError(msg)
    for state in ffds._states:
        data = ffds.load_partition(0, state).to_pandas()
        if isinstance(data, pd.DataFrame):
            datasets = preprocess_data_cross_silo(data, datasets, fairness_level, state)
    return datasets


def preprocess_data_cross_silo(
    data1: pd.DataFrame, datasets: list, fairness_level: str, state: str
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """
    Preprocess data for cross-silo federated learning by splitting into train/test and extracting sensitive features.

    Parameters
    ----------
    data1 : pd.DataFrame
        The input DataFrame for a specific state/silo.
    datasets : list
        Accumulator list of processed datasets.
    fairness_level : str
        The fairness level (e.g., "attribute").
    state : str
        The identifier for the state/silo.

    Returns
    -------
    list
        Updated list of datasets with the processed data tuple appended.

    """
    target1 = data1["PINCP"]
    data1.drop(inplace=True, columns=["PINCP"])
    if fairness_level == "attribute":
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            data1, target1, test_size=0.2, random_state=42, stratify=data1[["SEX", "RAC1P"]]
        )
    else:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state=42)
    sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
    X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
    X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
    datasets.append((state, X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))
    return datasets


def create_cross_device_data(
    fairness_level: Literal["attribute", "value", "attribute-value"], split_number: int, path: PathLike | str
) -> None:
    """
    Generate, evaluate, and save cross-device datasets based on previously generated cross-silo data.
    """
    path_str = str(path)
    datasets_all = _load_silo_datasets(fairness_level, split_number, path_str)

    df, fig = evaluate_models_on_datasets(datasets_all, n_jobs=3, fairness_level=fairness_level)
    df.to_csv(f"{path_str}data_stats/crossdevice_{fairness_level}.csv", index=False)

    states = _filter_states_by_fairness(df, fairness_level)

    for state in states:
        data1 = pd.read_csv(Path(f"{path_str}data/cross-device-{fairness_level}/{state}.csv"))
        data1.to_csv(Path(f"{path_str}data/cross_device_{fairness_level}_final/{state}.csv"))

    df = df[df["dataset"].isin(states)]
    df.to_csv(Path(f"{path_str}data/cross_device_{fairness_level}_final/model_perf_DP.csv"), index=False)


def _load_silo_datasets(fairness_level, split_number, path_str):
    datasets_all = []
    data_dir_path = Path(f"{path_str}data/cross_silo_{fairness_level}_final")
    for file_path in data_dir_path.iterdir():
        file = file_path.name
        if file in ["model_perf_DP.csv", "crosssilo_value_modifications.csv", "unfairness_distribution_DP.csv"]:
            continue
        data1 = pd.read_csv(Path(f"{path_str}data/cross_silo_{fairness_level}_final/{file}"))
        datasets = preprocess_datasets(file, data1, path_str, split_number, fairness_level)
        datasets_all.extend(datasets)
    return datasets_all


def _filter_states_by_fairness(df, fairness_level):
    if fairness_level == "attribute":
        return _filter_attribute_fairness(df)
    return _filter_value_fairness(df)


def _filter_attribute_fairness(df):
    states = []
    for entry in df["dataset"].unique():
        df_entry = df[(df["dataset"] == entry) & df["model"].isin(["XGBoost", "LogisticRegression"])]
        race_dp, sex_dp = df_entry["DP_RACE"].values, df_entry["DP_SEX"].values
        count = sum(1 for i in range(len(df_entry)) if sex_dp[i] > race_dp[i])

        if count == 2:
            if np.min(sex_dp) > 0.09:
                states.append(entry)
        elif count == 0 and 0.175 > np.min(race_dp) > 0.12:
            states.append(entry)
    return states


def _filter_value_fairness(df):
    states = []
    for entry in df["dataset"].unique():
        df_entry = df[(df["dataset"] == entry) & df["model"].isin(["XGBoost", "LogisticRegression"])]
        if (
            df_entry["value_DP_RACE"].values[0][-3:-2] == df_entry["value_DP_RACE"].values[1][-3:-2]
            and np.min(df_entry["DP_RACE"].values) > 0.09
        ):
            states.append(entry)
    return states


def preprocess_datasets(
    file: str, data1: pd.DataFrame, path: str, split_number: int = 6, fairness_level: str = "attribute"
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """
    Split a dataset into multiple parts and preprocess each for cross-device evaluation.

    Parameters
    ----------
    file : str
        Filename or identifier for the dataset.
    data1 : pd.DataFrame
        Input DataFrame.
    path : str
        Base path.
    split_number : int, default=6
        Number of splits.
    fairness_level : str, default="attribute"
        Fairness level.

    Returns
    -------
    list
        List of processed dataset tuples.

    """
    split_datasets = split_df(data1, split_number)
    datasets = []
    for i in range(len(split_datasets)):
        data1 = split_datasets[i]
        data1.to_csv(f"{path}data/cross-device-{fairness_level}/{file[:2]}_{i}.csv")
        target1 = data1["PINCP"]
        data1.drop(inplace=True, columns=["PINCP"])
        if fairness_level == "attribute":
            X_train1, X_test1, y_train1, y_test1 = train_test_split(
                data1, target1, test_size=0.2, random_state=42, stratify=data1[["SEX", "RAC1P"]]
            )
        else:
            X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state=42)

        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append((f"{file[:2]}_{i}", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))
    return datasets
