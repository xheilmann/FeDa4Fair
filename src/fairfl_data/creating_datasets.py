import os
from statistics import LinearRegression


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from FairFederatedDataset import FairFederatedDataset
from evaluation import evaluate_models_on_datasets, local_client_fairness_plot
from sklearn.model_selection import train_test_split

#example mapping parameter:
mapping =  {"MAR": { 3:2, 4:2, 5:2}, "RAC1P": {8:2, 7:2, 9:2, 6:2, 5:2, 4:2, 3:2}}

def split_df(df):
    if len(df) % 2 != 0:
       df = df.iloc[:-1, :]
    df1, df2 = np.array_split(df, 2)
    return df1, df2

def create_cross_silo_data_att():
    '''
    ffds = FairFederatedDataset(dataset="ACSIncome",  fl_setting=None, partitioners=None,
                                fairness_metric="DP", fairness_level="attribute",
                                mapping=mapping, path="/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_silo_attribute_final")

    datasets= []
    for state in ffds._states:

        data1 = ffds.load_partition(0, state).to_pandas()
        target1 = data1["PINCP"]
        data1.drop(inplace=True, columns=["PINCP"])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state =42 , stratify= data1[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace= True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append((state, X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))


    df, fig = evaluate_models_on_datasets(datasets, n_jobs=3)

    df.to_csv( f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crosssilo_attribute_0.0.csv", index=False)
    print(df)
    '''

    for dr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:

        partitioners = {}
        states = []
        modification_dict ={}
        #df = pd.read_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crosssilo_attribute_{np.round(dr-0.1, 2)}.csv")
        df = pd.read_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_silo_attribute_final/model_perf.csv")


        for entry in df["dataset"].unique():
            count = 0
            df_entry = df[df["dataset"] == entry]
            df_entry = df_entry[df_entry["model"].isin(["XGBoost", "LogisticRegression"])]
            for i in range(len(df_entry.values)):
                race_dp = df_entry["DP_RACE"].values[i]
                sex_dp = df_entry["DP_SEX"].values[i]
                if sex_dp > race_dp :
                    count += 1
            if count ==2:
                min = np.min(df_entry["DP_SEX"].values)
                print(entry, count, "SEX", min, np.min(df_entry["DP_RACE"].values))
                if min < 0.09:
                    modification_dict[entry] = {
                    "SEX":
                        {
                            "drop_rate": dr,
                            "flip_rate": 0,
                            "value": 2,
                            "attribute": None,
                            "attribute_value": None,
                        },

                }
                    states.append(entry)
                    partitioners[entry] = 1

            elif count ==0 :
                min = np.min(df_entry["DP_RACE"].values)
                print(entry, count, "RACE", min,  np.min(df_entry["DP_SEX"].values))
                if min < 0.09:
                    modification_dict[entry] = {
                    "RAC1P":
                        {
                            "drop_rate": dr,
                            "flip_rate": 0,
                            "value": 2,
                            "attribute": None,
                            "attribute_value": None,
                        },

                }
                    states.append(entry)
                    partitioners[entry] = 1
            else:
                print(entry, count)
                modification_dict[entry] = {
                    "SEX":
                        {
                            "drop_rate":dr,
                            "flip_rate": 0,
                            "value": 2,
                            "attribute": None,
                            "attribute_value": None,
                        },

                }
                states.append(entry)
                partitioners[entry] = 1


        ffds = FairFederatedDataset(dataset="ACSIncome",  fl_setting=None, partitioners=partitioners, states = states,
                                    fairness_metric="DP", fairness_level="attribute", modification_dict=modification_dict,
                                    mapping=mapping, path="/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_silo_attribute_final")

        datasets= []
        for state in ffds._states:

            data1 = ffds.load_partition(0, state).to_pandas()
            target1 = data1["PINCP"]
            data1.drop(inplace=True, columns=["PINCP"])
            X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state =42 , stratify= data1[["SEX", "RAC1P"]])
            sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
            X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
            X_test1.drop(inplace= True, columns=["MAR", "SEX", "RAC1P"])
            datasets.append((state, X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))


        df, fig = evaluate_models_on_datasets(datasets, n_jobs=3)

        df.to_csv( f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crosssilo_attribute_{dr}.csv", index=False)
        print(df)

def create_cross_device_data_att():
    '''
    datasets =[]
    dir = os.listdir("/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_silo_attribute_final")
    for file in dir:

        data1 = pd.read_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_silo_attribute_final/{file}")
        data1, data2 = split_df(data1)
        data1.to_csv((f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_0.csv"))
        data2.to_csv((f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_1.csv"))
        target1 = data1["PINCP"]
        data1.drop(inplace=True, columns=["PINCP"])

        X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state =42 , stratify= data1[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace= True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append((f"{file[:2]}_0", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))


        target1 = data2["PINCP"]
        data2.drop(inplace=True, columns=["PINCP"])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data2, target1, test_size=0.2, random_state=42,
                                                                stratify=data1[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append((f"{file[:2]}_1", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))





    df, fig = evaluate_models_on_datasets(datasets, n_jobs=3)

    df.to_csv( f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crossdevice_attribute_0.0.csv", index=False)
    print(df)

    '''
    for dr in [ 0.5, 0.6]:

        partitioners = {}
        states = []
        modification_dict = {}
        df = pd.read_csv(
            f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crossdevice_attribute_{np.round(dr - 0.1, 2)}.csv")


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
                        "SEX":
                            {
                                "drop_rate": dr,
                                "flip_rate": 0,
                                "value": 2,
                                "attribute": None,
                                "attribute_value": None,
                            },

                    }
                    states.append(entry[:2])


            elif count == 0:
                min = np.min(df_entry["DP_RACE"].values)
                print(entry, count, "RACE", min, np.min(df_entry["DP_SEX"].values))
                if min < 0.09:
                    modification_dict[entry] = {
                        "RAC1P":
                            {
                                "drop_rate": dr,
                                "flip_rate": 0,
                                "value": 2,
                                "attribute": None,
                                "attribute_value": None,
                            },

                    }
                    states.append(entry[:2])

            else:
                print(entry, count)
                modification_dict[entry] = {
                    "SEX":
                        {
                            "drop_rate": dr,
                            "flip_rate": 0,
                            "value": 2,
                            "attribute": None,
                            "attribute_value": None,
                        },

                }
                states.append(entry[:2])

        states = list(set(states))
        partitioners= {e:2 for e in states}
        ffds = FairFederatedDataset(dataset="ACSIncome", fl_setting=None, partitioners=partitioners, states=states,
                                    fairness_metric="DP", fairness_level="attribute",
                                    modification_dict=modification_dict,
                                    mapping=mapping,
                                    path="/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute")

        datasets = []
        for state in ffds._states:
            data1 = ffds.load_partition(0, state).to_pandas()
            target1 = data1["PINCP"]
            data1.drop(inplace=True, columns=["PINCP"])
            X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state=42,
                                                                    stratify=data1[["SEX", "RAC1P"]])
            sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
            X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
            X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
            datasets.append((f"{state}_0", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

            data1 = ffds.load_partition(1, state).to_pandas()
            target1 = data1["PINCP"]
            data1.drop(inplace=True, columns=["PINCP"])
            X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size=0.2, random_state=42,
                                                                    stratify=data1[["SEX", "RAC1P"]])
            sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
            X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
            X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
            datasets.append((f"{state}_1", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

        df, fig = evaluate_models_on_datasets(datasets, n_jobs=3)

        df.to_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crossdevice_attribute_{dr}.csv",
                  index=False)
        print(df)

#create_cross_device_data_att()
create_cross_silo_data_att()


