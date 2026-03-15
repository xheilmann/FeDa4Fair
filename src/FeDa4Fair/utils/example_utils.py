# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for examples and simulation."""

import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from FeDa4Fair.metrics.fairness import _compute_fairness


def pre_process_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the income dataset to make it ready for the simulation.

    In this function we consider "SEX" as the sensitive value and "PINCP" as the target value.

    Args:
        df: the raw data

    Returns:
        Returns a pre-processed pandas dataframe.

    """
    categorical_columns = ["COW", "SCHL"]
    continuous_columns = ["AGEP", "WKHP", "OCCP", "POBP", "RELP"]

    # convert the columns to one-hot encoding
    pre_df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

    # normalize the continuous columns between 0 and 1
    for col in continuous_columns:
        pre_df[col] = (pre_df[col] - pre_df[col].min()) / (pre_df[col].max() - pre_df[col].min())

    return pd.DataFrame(pre_df)


def pre_process_single_datasets(df: pd.DataFrame) -> tuple[list, list, list, list, list]:
    """Pre-process a single dataset for evaluation."""
    target_attributes = df[">50K"]
    sensitive_attributes = df["SEX"]
    second_sensitive_attributes = df["MAR"]
    third_sensitive_attributes = df["RAC1P"]
    third_sensitive_attributes = third_sensitive_attributes.astype(int)
    target_attributes = target_attributes.astype(int)

    sensitive_attributes = [1 if item == 1 else 0 for item in sensitive_attributes]
    second_sensitive_attributes = [1 if item == 1 else 0 for item in second_sensitive_attributes]
    third_sensitive_attributes = [1 if item == 1 else 0 for item in third_sensitive_attributes]

    features_df = df.drop([">50K"], axis=1)

    # convert the labels and groups to dataframes
    label = pd.DataFrame(target_attributes)
    group = pd.DataFrame(sensitive_attributes)
    second_group = pd.DataFrame(second_sensitive_attributes)
    third_group = pd.DataFrame(third_sensitive_attributes)

    if not (len(features_df) == len(label) == len(group) == len(second_group)):
        msg = "Lengths of dataframes do not match"
        raise ValueError(msg)

    dataframes = [features_df.to_numpy()]
    labels = [label.to_numpy()]
    groups = [group.to_numpy()]
    second_groups = [second_group.to_numpy()]
    third_groups = [third_group.to_numpy()]
    return dataframes, labels, groups, second_groups, third_groups


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""

    def __init__(self, x: Any, z: Any, w: Any, y: Any):
        """
        Initialize the custom dataset.

        Args:
            x: features
            z: sensitive attribute 1
            w: sensitive attribute 2
            y: targets

        """
        self.samples = x
        self.sensitive_features = z
        self.sensitive_features_2 = w
        self.targets = y
        self.indexes = list(range(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x_sample = self.samples[index]
        z_sample = self.sensitive_features[index]
        w_sample = self.sensitive_features_2[index]
        y_sample = self.targets[index]

        return x_sample, z_sample, w_sample, y_sample

    def shuffle(self):
        """Shuffle the dataset."""
        rng = np.random.default_rng()
        rng.shuffle(self.indexes)
        self.samples = [self.samples[i] for i in self.indexes]
        self.sensitive_features = [self.sensitive_features[i] for i in self.indexes]
        self.sensitive_features_2 = [self.sensitive_features_2[i] for i in self.indexes]
        self.targets = [self.targets[i] for i in self.indexes]


def pre_process_dataset_for_fl(states: list[str], ids: list[int], ffds: Any) -> None:
    """Pre-process dataset for Federated Learning simulation."""
    partitions = []
    partitions_names = []
    for state in states:
        for p_id in ids:
            partitions_names.append(f"{state}_{p_id}")

            partition = ffds.load_partition(split=state, partition_id=p_id)
            train_part, test_part = train_test_split(partition.to_pandas(), test_size=0.2)
            partitions.append(pd.DataFrame(train_part))
            partitions.append(pd.DataFrame(test_part))

    concatenated_df = pd.concat(partitions, ignore_index=True)
    concatenated_df["PINCP"] = [1 if item else 0 for item in concatenated_df["PINCP"]]

    # rename the column PINCP to >50K
    concatenated_df = concatenated_df.rename(columns={"PINCP": ">50K"})

    # Apply one-hot encoding
    pre_processed_df = pre_process_income(concatenated_df)

    split_dfs = []
    start_idx = 0
    for df_part in partitions:
        end_idx = start_idx + len(df_part)
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
            _train_third_groups,
        ) = pre_process_single_datasets(train_state)
        (
            test_data,
            test_labels,
            test_groups,
            test_second_groups,
            _test_third_groups,
        ) = pre_process_single_datasets(test_state)

        path_dir = Path(f"{folder}/federated/{index // 2}")
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)

        json_file = dict(enumerate(partitions_names))
        with Path(f"{folder}/federated/partitions_names.json").open("w") as f:
            json.dump(json_file, f)

        # save train
        custom_dataset = TabularDataset(
            x=np.hstack((train_data[0], np.ones((train_data[0].shape[0], 1)))).astype(np.float32),
            z=[item.item() for item in train_groups[0]],
            w=[item.item() for item in train_second_groups[0]],
            y=[item.item() for item in train_labels[0]],
        )

        torch.save(custom_dataset, f"{folder}/federated/{index // 2}/train.pt")

        # save test
        custom_dataset = TabularDataset(
            x=np.hstack((test_data[0], np.ones((test_data[0].shape[0], 1)))).astype(np.float32),
            z=[item.item() for item in test_groups[0]],
            w=[item.item() for item in test_second_groups[0]],
            y=[item.item() for item in test_labels[0]],
        )
        torch.save(custom_dataset, f"{folder}/federated/{index // 2}/test.pt")


class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size=11, output_size=2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.layer1(x.float())


def _run_evaluation_loop(
    net,
    testloader,
    device,
    sensitive_attrs: dict[str, int],
) -> tuple[float, float, dict]:
    """
    Run the evaluation loop and compute fairness metrics.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network to evaluate.
    testloader : DataLoader
        DataLoader whose batches expose sensitive attributes as positional elements.
    device : str | torch.device
        Device to run inference on.
    sensitive_attrs : dict[str, int]
        Mapping of attribute name to the batch-tuple position that holds it.
        For example ``{"SEX": 1, "MAR": 2}`` means ``batch[1]`` is SEX and
        ``batch[2]`` is MAR (0 is always images, last is always labels).

    Returns
    -------
    tuple[float, float, dict]
        ``(total_loss, accuracy, unfairness_dict)`` where *unfairness_dict* has
        keys ``<attr>_DP`` and ``<attr>_EO`` for each attribute.

    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    net.to(device)
    net.eval()

    # Buffers: one list per sensitive attribute + ground-truth & predictions
    sens_buffers: dict[str, list] = {name: [] for name in sensitive_attrs}
    true_y: list = []
    preds: list = []

    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[-1].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            true_y.extend(labels.cpu())
            preds.extend(predicted.cpu())
            for attr_name, batch_idx in sensitive_attrs.items():
                sens_buffers[attr_name].extend(batch[batch_idx])

    sf_data = pd.DataFrame({name: [int(v) for v in vals] for name, vals in sens_buffers.items()})

    unfairness_dict: dict = {}
    for attr_name in sensitive_attrs:
        for metric in ("DP", "EO"):
            unfairness_dict[f"{attr_name}_{metric}"] = _compute_fairness(
                y_true=true_y,
                y_pred=preds,
                sf_data=sf_data,
                fairness_metric=metric,
                sens_att=attr_name,
                size_unit="value",
            )

    accuracy = correct / len(testloader.dataset)
    return total_loss, accuracy, unfairness_dict


def train(net, trainloader, optimizer, device="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    for batch in trainloader:
        images, _, _, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()


def evaluate_tabular(net, testloader, device):
    """
    Validate the network on the entire test set.

    Batch convention: (images, sex, mar, labels)
    """
    # batch position 1 = SEX, position 2 = MAR
    return _run_evaluation_loop(
        net=net,
        testloader=testloader,
        device=device,
        sensitive_attrs={"SEX": 1, "MAR": 2},
    )


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate custom metrics."""
    # Scale metrics by number of examples and sum them up
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""
    params_dict = zip(model.state_dict().keys(), parameters, strict=False)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class ImageDataset(Dataset):
    """
    PyTorch Dataset for image data from Hugging Face datasets.
    """

    def __init__(self, hf_dataset, transform=None, label_key="label", sensitive_key="sensitive"):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_key = label_key
        self.sensitive_key = sensitive_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"]
        label = item[self.label_key]

        sensitive = item.get(self.sensitive_key, 0)

        if self.transform:
            image = self.transform(image)

        return image, sensitive, sensitive, label


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        images: list,
        labels: list,
        sensitive_attributes: list,
        transform: Callable | None = None,
    ) -> None:
        """
        Initialization of the dataset.

        Args:
        ----
            images (list): List of images.
            labels (list): List of labels.
            sensitive_attributes (list): List of sensitive attributes.
            transform (Callable | None, optional): Transformation to apply to the images. Defaults to None.

        """
        smiling_dict = {False: 0, True: 1}
        targets = [smiling_dict[item] for item in labels]
        self.targets = targets
        self.sensitive_attributes = [smiling_dict[item] for item in sensitive_attributes]
        self.samples = images
        self.n_samples = len(images)
        self.transform = transform
        self.indexes = range(len(self.samples))

    def __getitem__(self, index: int):
        """
        Returns a sample from the dataset.

        Args:
            index (int): index of the sample we want to retrieve

        Returns:
        -------
            _type_: sample we want to retrieve

        """
        img = self.samples[index]

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.sensitive_attributes[index],
            self.targets[index],
        )

    def __len__(self) -> int:
        """
        This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset

        """
        return self.n_samples


class SimpleCNN(nn.Module):
    """
    A simple CNN for image classification (e.g., for CelebA/MNIST).
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Assuming 64x64 input images -> 64/2/2 = 16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def evaluate_image(net, testloader, device, sensitive_attribute_name="sensitive"):
    """
    Validate the network on the entire test set for image data.

    Batch convention: (images, sensitive, labels)
    """
    # batch position 1 = sensitive attribute, position -1 = labels (handled by _run_evaluation_loop)
    return _run_evaluation_loop(
        net=net,
        testloader=testloader,
        device=device,
        sensitive_attrs={sensitive_attribute_name: 1},
    )


def get_default_image_transform(image_size=(64, 64)):
    """Get default transforms for image datasets."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def train_celeba(net, trainloader, optimizer, device="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    for batch in trainloader:
        images, _, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()


def evaluate_celeba(net, testloader, device):
    """
    Validate the network on the entire test set.

    Batch convention: (images, sex, labels)
    """
    # batch position 1 = SEX, position -1 = labels (handled by _run_evaluation_loop)
    return _run_evaluation_loop(
        net=net,
        testloader=testloader,
        device=device,
        sensitive_attrs={"SEX": 1},
    )
