import json
import os
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairness_computation import _compute_fairness
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


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

    categorical_columns = ["COW", "SCHL"]  # , "RAC1P"]
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

    second_sensitive_attributes = [1 if item == 1 else 0 for item in second_sensitive_attributes]

    third_sensitive_attributes = [1 if item == 1 else 0 for item in third_sensitive_attributes]

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


class TabularDataset(Dataset):
    def __init__(self, x, z, w, y):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.
        """
        self.samples = x
        self.sensitive_features = z
        self.sensitive_features_2 = w
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.
        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        w_sample = self.sensitive_features_2[idx]
        y_sample = self.targets[idx]

        return x_sample, z_sample, w_sample, y_sample

    def shuffle(self):
        """
        Shuffle the dataset.
        """
        self.indexes = list(self.indexes)
        np.random.shuffle(self.indexes)
        self.samples = [self.samples[i] for i in self.indexes]
        self.sensitive_features = [self.sensitive_features[i] for i in self.indexes]
        self.sensitive_features_2 = [self.sensitive_features_2[i] for i in self.indexes]
        self.targets = [self.targets[i] for i in self.indexes]
        self.indexes = range(len(self.samples))


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

        if not os.path.exists(f"{folder}/federated/{index // 2}"):
            os.makedirs(f"{folder}/federated/{index // 2}")
            # save partitions_names
        json_file = {index: data for index, data in enumerate(partitions_names)}
        with open(f"{folder}/federated/partitions_names.json", "w") as f:
            json.dump(json_file, f)

        # save train
        custom_dataset = TabularDataset(
            x=np.hstack((train_data[0], np.ones((train_data[0].shape[0], 1)))).astype(np.float32),
            z=[item.item() for item in train_groups[0]],  # .astype(np.float32),
            w=[item.item() for item in train_second_groups[0]],  # .astype(np.float32),
            y=[item.item() for item in train_labels[0]],  # .astype(np.float32),
        )

        torch.save(
            custom_dataset,
            f"{folder}/federated/{index // 2}/train.pt",
        )
        # save test
        custom_dataset = TabularDataset(
            x=np.hstack((test_data[0], np.ones((test_data[0].shape[0], 1)))).astype(np.float32),
            z=[item.item() for item in test_groups[0]],  # .astype(np.float32),
            w=[item.item() for item in test_second_groups[0]],  # .astype(np.float32),
            y=[item.item() for item in test_labels[0]],  # .astype(np.float32),
        )
        torch.save(
            custom_dataset,
            f"{folder}/federated/{index // 2}/test.pt",
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
        images, _, _, labels = batch
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
    sex_list = []
    mar_list = []
    true_y = []
    predictions = []
    with torch.no_grad():
        for batch in testloader:
            images, sex, mar, labels = batch
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            sex_list.extend(sex)
            mar_list.extend(mar)
            true_y.extend(labels)
            predictions.extend(predicted)

    sf_data = pd.DataFrame(
        {
            "SEX": [int(item) for item in sex_list],
            "MAR": [int(item) for item in mar_list],
        }
    )

    unfairness_dict = {}

    unfairness_dict["MAR_DP"] = _compute_fairness(
        y_true=true_y,
        y_pred=predictions,
        sf_data=sf_data,
        fairness_metric="DP",
        sens_att=["MAR"],
        size_unit="value",
    )
    unfairness_dict["SEX_DP"] = _compute_fairness(
        y_true=true_y,
        y_pred=predictions,
        sf_data=sf_data,
        fairness_metric="DP",
        sens_att=["SEX"],
        size_unit="value",
    )
    unfairness_dict["MAR_EO"] = _compute_fairness(
        y_true=true_y,
        y_pred=predictions,
        sf_data=sf_data,
        fairness_metric="EO",
        sens_att=["MAR"],
        size_unit="value",
    )
    unfairness_dict["SEX_EO"] = _compute_fairness(
        y_true=true_y,
        y_pred=predictions,
        sf_data=sf_data,
        fairness_metric="EO",
        sens_att=["SEX"],
        size_unit="value",
    )

    accuracy = correct / len(testloader.dataset)

    return loss, accuracy, unfairness_dict


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


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
