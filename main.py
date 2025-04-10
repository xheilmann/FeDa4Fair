

from FairFederatedDataset import FairFederatedDataset

#example binary_mapping parameter:
#binary_mapping: {"MAR": {2:1, 3:1, 4:1, 5:1}, "RAC1P": {8:6, 7:6, 9:6}}

for name in ["ACSIncome"]:
    for train_test_split in [None]:
        for fairness in ["DP" ]:
            for in_fairness in [ "attribute", "value"]:
                ffds = FairFederatedDataset(dataset="ACSIncome", states=["CT", "AK", "FL", "CO", "AZ", "AR"],
                                partitioners={"CT":5, "AK":5, "FL":5, "CO":5, "AZ":5, "AR":5}, train_test_split=train_test_split,
                                        fairness_metric=fairness, individual_fairness=in_fairness)


                if train_test_split == None:
                    split = ffds.load_split("FL")
                if train_test_split == "cross-silo" or train_test_split == "cross-device":
                    split = ffds.load_split("CT_train")
                    print(split)
                #partition = ffds.load_partition(0, "CT_val")
                if train_test_split == "cross-device":
                    test = ffds.load_split("test")
                    print(test)

#todo test the evaluate function with model
