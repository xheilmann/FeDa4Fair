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
    if len(df) % 6 != 0:
       df = df.iloc[:-(len(df)%6), :]
    df1, df2, df3, df4, df_5, df_6 = np.array_split(df, 6)
    return df1, df2, df3, df4, df_5, df_6

def create_cross_silo_data_att():
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

    for dr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:

        partitioners = {}
        states = []
        modification_dict ={}
        df = pd.read_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crosssilo_attribute_{np.round(dr-0.1, 2)}.csv")


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
        if file == "model_perf.csv":
            continue

        data1 = pd.read_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_silo_attribute_final/{file}")
        data1, data2, data3, data4, data5, data6 = split_df(data1)
        data1.to_csv((f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_0.csv"))
        data2.to_csv((f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_1.csv"))
        data3.to_csv(
            (f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_2.csv"))
        data4.to_csv(
           (f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_3.csv"))
        data5.to_csv(
            (f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_4.csv"))
        data6.to_csv(
            (f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{file[:2]}_5.csv"))


        print(file)

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
                                                                stratify=data2[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append((f"{file[:2]}_1", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

        target1 = data3["PINCP"]
        data3.drop(inplace=True, columns=["PINCP"])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data3, target1, test_size=0.2, random_state=42,
                                                                stratify=data3[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append(
            (f"{file[:2]}_2", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

        target1 = data4["PINCP"]
        data4.drop(inplace=True, columns=["PINCP"])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data4, target1, test_size=0.2, random_state=42,
                                                                stratify=data4[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append(
            (f"{file[:2]}_3", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

        target1 = data5["PINCP"]
        data5.drop(inplace=True, columns=["PINCP"])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data5, target1, test_size=0.2, random_state=42,
                                                                stratify=data5[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append(
            (f"{file[:2]}_4", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

        target1 = data6["PINCP"]
        data6.drop(inplace=True, columns=["PINCP"])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data6, target1, test_size=0.2, random_state=42,
                                                                stratify=data6[["SEX", "RAC1P"]])
        sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}
        X_train1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        X_test1.drop(inplace=True, columns=["MAR", "SEX", "RAC1P"])
        datasets.append(
            (f"{file[:2]}_5", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1))

    df, fig = evaluate_models_on_datasets(datasets, n_jobs=3)

    df.to_csv( f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crossdevice_attribute.csv", index=False)
    print(df)


    '''


    states = []
    df = pd.read_csv(
        f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data_stats/crossdevice_attribute.csv")
    df = pd.read_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_device_attribute_final/model_perf.csv")
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
            if 0.175>min > 0.12:
                states.append(entry)
                r_count +=1

        else:
            print(entry, count)

    print(r_count, s_count)

    datasets = []

    for state in states:
        data1 = pd.read_csv(
            f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross-device-attribute/{state}.csv")
        data1.to_csv(
            (f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_device_attribute_final/{state}.csv"))


    df = df[df["dataset"].isin(states)]

    df.to_csv(f"/home/heilmann/Dokumente/fairFL-data/src/fairfl_data/data/cross_device_attribute_final/model_perf.csv",
              index=False)
    print(df)


create_cross_device_data_att()



