from FairFederatedDataset import FairFederatedDataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


for name in ["ACSIncome"]:
    for train_test_split in [None, "cross-silo", "cross-device"]:
        for fairness in ["EO", "DP"]:
            for in_fairness in ["attribute", "value"]:
                ffds = FairFederatedDataset(
                    dataset="ACSIncome",
                    states=["CT", "DE"],
                    partitioners={"CT": 2, "DE": 1},
                    train_test_split=train_test_split,
                    fairness_metric=fairness,
                    individual_fairness=in_fairness,
                )

                if train_test_split == "cross-silo" or train_test_split == "cross-device":
                    split = ffds.load_split("CT_train")
                    print(split)
                # partition = ffds.load_partition(0, "CT_val")
                if train_test_split == "cross-device":
                    test = ffds.load_split("test")
                    print(test)

# todo test the evaluate function with model
