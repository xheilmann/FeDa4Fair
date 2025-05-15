import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from FairFederatedDataset import FairFederatedDataset
from evaluation import evaluate_models_on_datasets
from sklearn.model_selection import train_test_split

# mapping parameter for attribute unfairness
# mapping =  {"MAR": { 3:2, 4:2, 5:2}, "RAC1P": {8:2, 7:2, 9:2, 6:2, 5:2, 4:2, 3:2}}
# mapping parameter for value unfairness
mapping = {"MAR": {3: 2, 4: 2, 5: 2}, "RAC1P": {8: 5, 7: 5, 9: 5, 6: 3, 5: 4, 3: 4}}


def split_df(df, split_number):
    a = np.array_split(df, split_number)
    return a


def create_cross_silo_data(fairness_level, path):
    ffds = FairFederatedDataset(
        dataset="ACSIncome",
        fl_setting=None,
        partitioners=None,
        fairness_metric="DP",
        fairness_level=fairness_level,
        mapping=mapping,
        path=f"{path}data/cross_silo_{fairness_level}_final",
    )
    datasets = []
    for state in ffds._states:
        data1 = ffds.load_partition(0, state).to_pandas()
        datasets = preprocess_data_cross_silo(data1, datasets, fairness_level, state)

    df, fig = evaluate_models_on_datasets(datasets, n_jobs=3, fairness_level=fairness_level)
    df.to_csv(f"{path}data_stats/crosssilo_{fairness_level}_0.0.csv", index=False)
    print(df)

    all_modifications = []
    for dr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        partitioners = {}
        states = []
        modification_dict = {}
        df = pd.read_csv(f"{path}data_stats/crosssilo_{fairness_level}_{np.round(dr - 0.1, 2)}.csv")
        if fairness_level == "attribute":
            for entry in df["dataset"].unique():
                count = 0
                df_entry = df[df["dataset"] == entry]
                df_entry = df_entry[df_entry["model"].isin(["XGBoost", "LogisticRegression"])]
                for i in range(len(df_entry.values)):
                    race_dp = df_entry["DP_RACE"].values[i]
                    sex_dp = df_entry["DP_SEX"].values[i]
                    if sex_dp > race_dp:
                        count += 1
                if count == 2:
                    min = np.min(df_entry["DP_SEX"].values)
                    print(entry, count, "SEX", min, np.min(df_entry["DP_RACE"].values))
                    if min < 0.09:
                        modification_dict[entry] = {
                            "SEX": {
                                "drop_rate": dr,
                                "flip_rate": 0,
                                "value": 2,
                                "attribute": None,
                                "attribute_value": None,
                            },
                        }
                        states.append(entry)
                        partitioners[entry] = 1
                        all_modifications.append([entry, dr, "SEX", 2])
                elif count == 0:
                    min = np.min(df_entry["DP_RACE"].values)
                    print(entry, count, "RACE", min, np.min(df_entry["DP_SEX"].values))
                    if min < 0.09:
                        modification_dict[entry] = {
                            "RAC1P": {
                                "drop_rate": dr,
                                "flip_rate": 0,
                                "value": 2,
                                "attribute": None,
                                "attribute_value": None,
                            },
                        }
                        states.append(entry)
                        partitioners[entry] = 1
                        all_modifications.append([entry, dr, "RAC1P", 2])
                else:
                    print(entry, count)
                    modification_dict[entry] = {
                        "SEX": {
                            "drop_rate": dr,
                            "flip_rate": 0,
                            "value": 2,
                            "attribute": None,
                            "attribute_value": None,
                        },
                    }
                    states.append(entry)
                    partitioners[entry] = 1
                    all_modifications.append([entry, dr, "SEX", 2])
        else:
            count = [0, 0, 0, 0, 0]
            states_unfairness_distribution = {1: [], 2: [], 3: [], 4: [], 5: []}
            for entry in df["dataset"].unique():
                df_entry = df[df["dataset"] == entry]
                df_entry = df_entry[df_entry["model"].isin(["XGBoost", "LogisticRegression"])]
                value1 = df_entry["value_DP_RACE"].values[0][-3:-2]
                value2 = df_entry["value_DP_RACE"].values[1][-3:-2]
                if value1 == value2:
                    min = np.min(df_entry["DP_RACE"].values)
                    print(entry, value1, min)
                    count[int(value1) - 1] += 1
                    states_unfairness_distribution[int(value1)] = states_unfairness_distribution[int(value1)] + [entry]
                    if min < 0.09:
                        modification_dict[entry] = {
                            "RAC1P": {
                                "drop_rate": dr,
                                "flip_rate": 0,
                                "value": int(df_entry["value_DP_RACE"].values[0][:1]),
                                "attribute": None,
                                "attribute_value": None,
                            },
                        }
                        states.append(entry)
                        partitioners[entry] = 1
                        all_modifications.append([entry, dr, "RAC1P", int(df_entry["value_DP_RACE"].values[0][:1])])
                else:
                    print(entry)
                    modification_dict[entry] = {
                        "RAC1P": {
                            "drop_rate": dr,
                            "flip_rate": 0,
                            "value": int(df_entry["value_DP_RACE"].values[0][-3:-2]),
                            "attribute": None,
                            "attribute_value": None,
                        },
                    }
                    states.append(entry)
                    partitioners[entry] = 1
                    all_modifications.append([entry, dr, "RAC1P", int(df_entry["value_DP_RACE"].values[0][-3:-2])])

        print(count)
        print(states_unfairness_distribution)

        if len(states) <= 1:
            break
        ffds = FairFederatedDataset(
            dataset="ACSIncome",
            fl_setting=None,
            partitioners=partitioners,
            states=states,
            fairness_metric="DP",
            fairness_level=fairness_level,
            modification_dict=modification_dict,
            mapping=mapping,
            path=f"{path}data/cross_silo_{fairness_level}_final",
        )
        datasets = []
        for state in ffds._states:
            data1 = ffds.load_partition(0, state).to_pandas()
            datasets = preprocess_data_cross_silo(data1, datasets, fairness_level, state)

        df, fig = evaluate_models_on_datasets(datasets, n_jobs=3, fairness_level=fairness_level)
        df.to_csv(f"{path}data_stats/crosssilo_{fairness_level}_{dr}.csv", index=False)

        print(df)

    all_modifications_df = DataFrame(data=all_modifications, columns=["state", "drop_rate", "attribute", "value"])
    all_modifications_df.to_csv(f"{path}data_stats/crosssilo_{fairness_level}_modifications.csv", index=False)


def preprocess_data_cross_silo(data1, datasets, fairness_level, state):
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


def create_cross_device_data(fairness_level, split_number, path):
    datasets_all = []
    dir = os.listdir(f"{path}data/cross_silo_{fairness_level}_final")
    for file in dir:
        if file in ["model_perf_DP.csv", "crosssilo_value_modifications.csv", "unfairness_distribution_DP.csv"]:
            continue
        data1 = pd.read_csv(f"{path}data/cross_silo_{fairness_level}_final/{file}")
        datasets = preprocess_datasets(file, data1, path, split_number, fairness_level)
        for e in datasets:
            datasets_all.append(e)
    df, fig = evaluate_models_on_datasets(datasets_all, n_jobs=3, fairness_level=fairness_level)
    df.to_csv(f"{path}data_stats/crossdevice_{fairness_level}.csv", index=False)
    print(df)
    states = []

    if fairness_level == "attribute":
        s_count = 0
        r_count = 0
        for entry in df["dataset"].unique():
            count = 0
            df_entry = df[df["dataset"] == entry]
            df_entry = df_entry[df_entry["model"].isin(["XGBoost", "LogisticRegression"])]
            for i in range(len(df_entry.values)):
                race_dp = df_entry["DP_RACE"].values[i]
                sex_dp = df_entry["DP_SEX"].values[i]
                if sex_dp > race_dp:
                    count += 1
            if count == 2:
                min = np.min(df_entry["DP_SEX"].values)
                print(entry, count, "SEX", min, np.min(df_entry["DP_RACE"].values))
                if min > 0.09:
                    states.append(entry)
                    s_count += 1
            elif count == 0:
                min = np.min(df_entry["DP_RACE"].values)
                print(entry, count, "RACE", min, np.min(df_entry["DP_SEX"].values))
                if 0.175 > min > 0.12:
                    states.append(entry)
                    r_count += 1
            else:
                print(entry, count)
        print(r_count, s_count)
    else:
        count = [0, 0, 0, 0, 0]
        for entry in df["dataset"].unique():
            df_entry = df[df["dataset"] == entry]
            df_entry = df_entry[df_entry["model"].isin(["XGBoost", "LogisticRegression"])]
            value1 = df_entry["value_DP_RACE"].values[0][-3:-2]
            value2 = df_entry["value_DP_RACE"].values[1][-3:-2]
            if value1 == value2:
                min = np.min(df_entry["DP_RACE"].values)
                print(entry, value1, min)
                if min > 0.09:
                    states.append(entry)
                    count[int(value1) - 1] += 1
            else:
                print(entry)
        print(count)

    for state in states:
        data1 = pd.read_csv(f"{path}data/cross-device-{fairness_level}/{state}.csv")
        data1.to_csv((f"{path}data/cross_device_{fairness_level}_final/{state}.csv"))
    df = df[df["dataset"].isin(states)]
    df.to_csv(f"{path}data/cross_device_{fairness_level}_final/model_perf_DP.csv", index=False)
    print(df)


def preprocess_datasets(file, data1, path, split_number=6, fairness_level="attribute"):
    split_datasets = split_df(data1, split_number)
    datasets = []
    for i in range(len(split_datasets)):
        data1 = split_datasets[i]
        data1.to_csv((f"{path}data/cross-device-{fairness_level}/{file[:2]}_{i}.csv"))
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
