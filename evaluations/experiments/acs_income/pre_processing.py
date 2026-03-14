import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)


def pre_process_income(df):
    """
    Pre-process the income dataset to make it ready for the simulation
    In this function we consider "SEX" as the sensitive value and "PINCP" as the target value.

    Args:
        df: The pandas dataframe to be pre-processed.

    Returns:
        Returns a list of pre-processed data for each state, if multiple years are
        selected, the data are concatenated.
        We return three lists:
        - The first list contains a pandas dataframe of features for each state
        - The second list contains a pandas dataframe of labels for each state
        - The third list contains a pandas dataframe of groups for each state
        The values in the list are numpy array of the dataframes

    """
    categorical_columns = ["COW", "SCHL"]  # , "RAC1P"]
    continuous_columns = ["AGEP", "WKHP", "OCCP", "POBP", "RELP"]

    # get the target and sensitive attributes
    df[">50K"]
    df["SEX"]

    # convert the columns to one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

    # normalize the continuous columns between 0 and 1
    for col in continuous_columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return pd.DataFrame(df)


def pre_process_single_datasets(df):
    dataframe = pd.DataFrame()
    label = pd.DataFrame()
    group = pd.DataFrame()
    second_group = pd.DataFrame()
    third_group = pd.DataFrame()
    dataframes = []
    labels = []
    groups = []
    second_groups = []
    third_groups = []
    target_attributes = df[">50K"]
    sensitive_attributes = df["SEX"]
    second_sensitive_attributes = df["MAR"]

    third_sensitive_attributes = df["RAC1P"]
    third_sensitive_attributes = third_sensitive_attributes.astype(int)
    target_attributes = target_attributes.astype(int)

    sensitive_attributes = [1 if item == 1 else 0 for item in sensitive_attributes]

    second_sensitive_attributes = [1 if item == 1 else 0 for item in second_sensitive_attributes]

    df = df.drop([">50K"], axis=1)

    # concatenate the dataframes
    dataframe = pd.concat([dataframe, df])
    # remove RAC1P from dataframe

    # convert the labels and groups to dataframes
    label = pd.concat([label, pd.DataFrame(target_attributes)])
    group = pd.concat([group, pd.DataFrame(sensitive_attributes)])
    second_group = pd.concat([second_group, pd.DataFrame(second_sensitive_attributes)])
    third_group = pd.concat([third_group, pd.DataFrame(third_sensitive_attributes)])

    if not (len(dataframe) == len(label) == len(group) == len(second_group)):
        msg = "The lengths of the dataframes, labels, and groups do not match."
        raise ValueError(msg)
    dataframes.append(dataframe.to_numpy())
    labels.append(label.to_numpy())
    groups.append(group.to_numpy())
    second_groups.append(second_group.to_numpy())
    third_groups.append(third_group.to_numpy())
    return dataframes, labels, groups, second_groups, third_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross_silo",
        type=bool,
        help="Whether to pre-process the data for cross-silo or cross-device setting",
        default=False,
    )

    parser.add_argument("--folder_name", type=str, help="The folder name where the data is stored")

    args = parser.parse_args()
    cross_silo = args.cross_silo
    folder = Path(args.folder_name)

    unfair_dfs = []

    states = [
        "CT",
        "RI",
        "VT",
        "TX",
        "GA",
        "PR",
        "OH",
        "NE",
        "HI",
        "MO",
        "PA",
        "DE",
        "WV",
        "MD",
        "AZ",
        "LA",
        "WA",
        "TN",
        "MA",
        "NJ",
        "ME",
        "SC",
        "MI",
        "OK",
        "IL",
        "FL",
        "UT",
        "AK",
        "WI",
        "NH",
        "VA",
        "SD",
        "MS",
        "ND",
        "NC",
        "AL",
        "IA",
        "ID",
        "WY",
        "NV",
        "NM",
        "NY",
        "CA",
        "AR",
        "MN",
        "OR",
        "MT",
        "KY",
        "KS",
        "IN",
        "CO",
    ]

    partitions_names = []
    list_files = list(folder.iterdir())

    for state in states:
        partitions = set()
        for file in list_files:
            if file.name.endswith(".csv"):
                if not file.name.startswith(state):
                    continue
                partition = int(file.name.split("_")[-1].split(".")[0])
                if partition not in partitions:
                    partitions_names.append(f"{state}_{partition}")
                    partitions.add(partition)
                    try:
                        train = pd.read_csv(file)
                        # split the train csv into train and test
                        if cross_silo:
                            train, test = train_test_split(train, test_size=0.2, random_state=42)

                        unfair_dfs.append(train)
                        if cross_silo:
                            unfair_dfs.append(test)
                    except (pd.errors.ParserError, OSError) as e:
                        print(f"Error reading file {file.name}: {e}")
                        continue

    concatenated_df = pd.concat(unfair_dfs, ignore_index=True)
    concatenated_df["PINCP"] = [1 if item else 0 for item in concatenated_df["PINCP"]]

    # rename the column PINCP to >50K
    concatenated_df = concatenated_df.rename(columns={"PINCP": ">50K"})

    concatenated_df = concatenated_df.drop(["__index_level_0__"], axis=1)

    pre_processed_df = pre_process_income(concatenated_df)

    split_dfs = []
    start_idx = 0
    for df in unfair_dfs:
        end_idx = start_idx + len(df)
        split_dfs.append(pre_processed_df.iloc[start_idx:end_idx])
        start_idx = end_idx

    if cross_silo:
        print("Cross-silo setting")
        for index in range(0, len(split_dfs), 2):
            print("Processing partition:", index // 2)
            train_state = split_dfs[index]
            test_state = split_dfs[index + 1]
            print(len(train_state), len(test_state))
            (
                train_data,
                train_labels,
                train_groups,
                train_second_groups,
                train_third_groups,
            ) = pre_process_single_datasets(train_state)
            (
                test_data,
                test_labels,
                test_groups,
                test_second_groups,
                test_third_groups,
            ) = pre_process_single_datasets(test_state)

            print(index // 2, train_data[0].shape, test_data[0].shape)

            federated_path = folder / "FL_data" / "federated" / str(index // 2)
            if not federated_path.exists():
                federated_path.mkdir(parents=True)
                # save partitions_names
            json_file = dict(enumerate(partitions_names))
            with (folder / "FL_data" / "federated" / "partitions_names.json").open("w") as f:
                json.dump(json_file, f)
            np.save(
                federated_path / f"income_dataframes_{index // 2}_train.npy",
                train_data[0],
            )
            np.save(
                federated_path / f"income_labels_{index // 2}_train.npy",
                train_labels[0],
            )
            np.save(
                federated_path / f"income_groups_{index // 2}_train.npy",
                train_groups[0],
            )
            np.save(
                federated_path / f"income_second_groups_{index // 2}_train.npy",
                train_second_groups[0],
            )
            np.save(
                federated_path / f"income_third_groups_{index // 2}_train.npy",
                train_third_groups[0],
            )

            np.save(
                federated_path / f"income_dataframes_{index // 2}_test.npy",
                test_data[0],
            )
            np.save(
                federated_path / f"income_labels_{index // 2}_test.npy",
                test_labels[0],
            )
            np.save(
                federated_path / f"income_groups_{index // 2}_test.npy",
                test_groups[0],
            )
            np.save(
                federated_path / f"income_second_groups_{index // 2}_test.npy",
                test_second_groups[0],
            )
            np.save(
                federated_path / f"income_third_groups_{index // 2}_test.npy",
                test_third_groups[0],
            )
    else:
        print("Cross-device setting")
        for index in range(len(split_dfs)):
            print("Processing partition:", index)
            train_state = split_dfs[index]
            (
                train_data,
                train_labels,
                train_groups,
                train_second_groups,
                train_third_groups,
            ) = pre_process_single_datasets(train_state)

            print(index, train_data[0].shape)

            federated_path = folder / "FL_data" / "federated" / str(index)
            if not federated_path.exists():
                federated_path.mkdir(parents=True)
                # save partitions_names
            json_file = dict(enumerate(partitions_names))
            with (folder / "FL_data" / "federated" / "partitions_names.json").open("w") as f:
                json.dump(json_file, f)
            np.save(
                federated_path / f"income_dataframes_{index}_train.npy",
                train_data[0],
            )
            np.save(
                federated_path / f"income_labels_{index}_train.npy",
                train_labels[0],
            )
            np.save(
                federated_path / f"income_groups_{index}_train.npy",
                train_groups[0],
            )
            np.save(
                federated_path / f"income_second_groups_{index}_train.npy",
                train_second_groups[0],
            )
            np.save(
                federated_path / f"income_third_groups_{index}_train.npy",
                train_third_groups[0],
            )
