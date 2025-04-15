

from FairFederatedDataset import FairFederatedDataset
from evaluation import evaluate_models_on_datasets
from sklearn.model_selection import train_test_split

#example mapping parameter:
#mapping: {"MAR": {2:1, 3:1, 4:1, 5:1}, "RAC1P": {8:6, 7:6, 9:6}}

#example for modification dict (this is always after the mapping!
modification_dict = {
"CT": {
"MAR":
{
"drop_rate": 0.2,
"flip_rate": 0.1,
"value": 2,
"attribute": "SEX",
"attribute_value": 2,
},
"SEX":
{
"drop_rate": 0.3,
"flip_rate": 0.2,
"value": 2,
"attribute": None,
"attribute_value": None,
},
}}


for name in ["ACSIncome"]:
    for tt_split in [None]:
        for fairness in ["DP" ]:
            for in_fairness in [ "attribute", "value"]:
                ffds = FairFederatedDataset(dataset="ACSIncome", states=["CT", "AK"],
                                partitioners={"CT":5, "AK":5}, train_test_split=tt_split,
                                        fairness_metric=fairness, individual_fairness=in_fairness, modification_dict=modification_dict)


                if tt_split == None:
                    #split = ffds.load_split("FL")
                    data1 = ffds.load_partition(0, "CT").to_pandas()
                    target1 = data1["PINCP"]
                    data1.drop(inplace=True, columns=["PINCP"])
                    X_train1, X_test1, y_train1, y_test1 = train_test_split(data1, target1, test_size= 0.2)
                    sf_data1 = {"SEX": X_test1["SEX"].values, "RACE": X_test1["RAC1P"].values, "MAR": X_test1["MAR"].values}

                    data2 = ffds.load_partition(0, "AK").to_pandas()
                    target2 = data2["PINCP"]
                    data2.drop(inplace=True, columns=["PINCP"])
                    X_train2, X_test2, y_train2, y_test2 = train_test_split(data2, target2, test_size=0.2)
                    sf_data2 = {"SEX": X_test2["SEX"].values, "RACE": X_test2["RAC1P"].values,
                                "MAR": X_test2["MAR"].values}

                    df, fig = evaluate_models_on_datasets([("CT_0", X_train1.values, y_train1.values, X_test1.values, y_test1.values, sf_data1), ("AK_O", X_train2.values, y_train2.values, X_test2.values, y_test2.values, sf_data2)], n_jobs=2
                    )
                    print(df)
                    break
                if train_test_split == "cross-silo" or train_test_split == "cross-device":
                    split = ffds.load_split("CT_train")
                    print(split)
                #partition = ffds.load_partition(0, "CT_val")
                if train_test_split == "cross-device":
                    test = ffds.load_split("test")
                    print(test)

#todo test the evaluate function with model
