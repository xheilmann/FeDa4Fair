import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os 
import numpy as np

def pre_process_income(df):
    """
    Pre-process the income dataset to make it ready for the simulation
    In this function we consider "SEX" as the sensitive value and "PINCP" as the target value.

    Args:
        data: the raw data
        years_list: the list of years to be considered
        states_list: the list of states to be considered

    Returns:
        Returns a list of pre-processed data for each state, if multiple years are
        selected, the data are concatenated.
        We return three lists:
        - The first list contains a pandas dataframe of features for each state
        - The second list contains a pandas dataframe of labels for each state
        - The third list contains a pandas dataframe of groups for each state
        The values in the list are numpy array of the dataframes
    """

    categorical_columns = ["COW", "SCHL"] #, "RAC1P"]
    continuous_columns = ["AGEP", "WKHP", "OCCP", "POBP", "RELP"]

    # get the target and sensitive attributes
    target_attributes = df[">50K"]
    sensitive_attributes = df["SEX"]

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

    second_sensitive_attributes = [
        1 if item == 1 else 0 for item in second_sensitive_attributes
    ]

    third_sensitive_attributes = [
        1 if item == 1 else 0 for item in third_sensitive_attributes
    ]

    df = df.drop([">50K"], axis=1)
    # df.drop(['RAC1P_1.0', 'RAC1P_2.0'], axis=1, inplace=True)

    # concatenate the dataframes
    dataframe = pd.concat([dataframe, df])
    # remove RAC1P from dataframe

    # convert the labels and groups to dataframes
    label = pd.concat([label, pd.DataFrame(target_attributes)])
    group = pd.concat([group, pd.DataFrame(sensitive_attributes)])
    second_group = pd.concat([second_group, pd.DataFrame(second_sensitive_attributes)])
    third_group = pd.concat([third_group, pd.DataFrame(third_sensitive_attributes)])

    assert len(dataframe) == len(label) == len(group) == len(second_group)
    dataframes.append(dataframe.to_numpy())
    labels.append(label.to_numpy())
    groups.append(group.to_numpy())
    second_groups.append(second_group.to_numpy())
    third_groups.append(third_group.to_numpy())
    return dataframes, labels, groups, second_groups, third_groups

def pre_process_dataset_for_FL(states, ids, ffds):
    partitions = []
    partitions_names = []
    for state in states:
        for partition_ID in ids:
            # partition = partitioner.load_partition(partition_id=i)
            partitions_names.append(f"{state}_{partition_ID}")

            partition = ffds.load_partition(split=state, partition_id=partition_ID)
            train, test = train_test_split(partition.to_pandas(), test_size=0.2)
            partitions.append(pd.DataFrame(train))
            partitions.append(pd.DataFrame(test))


    concatenated_df = pd.concat(partitions, ignore_index=True)
    concatenated_df["PINCP"] = [1 if item == True else 0 for item in concatenated_df["PINCP"]]

    # rename the column PINCP to >50K
    concatenated_df.rename(columns={"PINCP": ">50K"}, inplace=True)

    # Apply one-hot encoding
    pre_processed_df = pre_process_income(concatenated_df)


    split_dfs = []
    start_idx = 0
    for df in partitions:
        end_idx = start_idx + len(df)

        split_dfs.append(pre_processed_df.iloc[start_idx:end_idx])
        start_idx = end_idx

    folder = "./data/"
    for index in range(0, len(split_dfs), 2):
        train_state = split_dfs[index]
        test_state = split_dfs[index + 1]
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

        if not os.path.exists(
            f"{folder}/federated/{index // 2}"
        ):
            os.makedirs(f"{folder}/federated/{index // 2}")
            # save partitions_names 
        json_file = {index:data for index, data in enumerate(partitions_names)}
        with open(f"{folder}/federated/partitions_names.json", "w") as f:
            json.dump(json_file, f)
        np.save(
            f"{folder}/federated/{index // 2}/income_dataframes_{index // 2}_train.npy",
            train_data[0],
        )
        np.save(
            f"{folder}/federated/{index // 2}/income_labels_{index // 2}_train.npy",
            train_labels[0],
        )
        
        np.save(
            f"{folder}/federated/{index // 2}/income_dataframes_{index // 2}_test.npy",
            test_data[0],
        )
        np.save(
            f"{folder}/federated/{index // 2}/income_labels_{index // 2}_test.npy",
            test_labels[0],
        )

class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size=11, output_size=2):
        super(LinearClassificationNet, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.layer1(x.float())
        return x


def train(net, trainloader, optimizer, device="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    for batch in trainloader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# Two auxhiliary functions to set and extract parameters of a model
def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]