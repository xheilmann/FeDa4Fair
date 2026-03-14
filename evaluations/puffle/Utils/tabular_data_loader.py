import json
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import figure
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from Utils.celeba import CelebaPreparedDataset
from Utils.dutch import TabularDataset
from Utils.utils import Utils

##############################################################################################################


def plot_distribution(distributions, title):
    counters = {}
    key_list = set()
    nodes_data = []
    for distribution in distributions:
        counter = Counter(distribution)
        nodes_data.append(counter)
        for key in counter:
            key_list.add(key)

    for node_data in nodes_data:
        for key in key_list:
            if key not in node_data:
                if key not in counters:
                    counters[key] = []
                counters[key].append(0)
            else:
                if key not in counters:
                    counters[key] = []
                counters[key].append(node_data[key])

    figure(figsize=(25, 8), dpi=80)
    indexes = np.arange(len(distributions))

    legend_values = []
    colors = ["blue", "green", "red", "purple", "orange", "yellow", "pink", "brown"]
    markers = ["+", "o", "x", "*", "v", "^", "<", ">"]
    for i, (key, value) in enumerate(counters.items()):
        plt.scatter(indexes, value, marker=markers[i], linewidths=4, s=100, color=colors[i])
        legend_values.append(key)
    plt.xticks(indexes, list(range(len(indexes))))
    plt.rcParams.update({"font.size": 16})
    plt.xlabel("Nodes")
    plt.ylabel("Samples")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.legend(legend_values)
    plt.savefig("distribution.png")


#########################################################################################


def get_tabular_data(
    dataset_name: str,
    num_sensitive_features: int,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    dataset_path=None,
    group_to_reduce: tuple | None = None,
    group_to_increment: tuple | None = None,
    number_of_samples_per_node: int | None = None,
    opposite_group_to_reduce: tuple | None = None,
    opposite_group_to_increment: tuple | None = None,
    opposite_ratio_unfairness: tuple | None = None,
    one_group_nodes: bool = False,
):
    x_data, z, y = get_tabular_numpy_dataset(
        dataset_name=dataset_name,
        num_sensitive_features=num_sensitive_features,
        dataset_path=dataset_path,
    )
    z = z[:, 0]
    print(f"Data shapes: x={x_data.shape}, y={y.shape}, z={z.shape}")
    # Prepare training data held by each client
    # Metadata is a list with 0 if the client is fair, 0 otherwise
    client_data, metadata = generate_clients_biased_data_mod(
        x_data=x_data,
        y=y,
        z=z,
        approach=approach,
        num_nodes=num_nodes,
        ratio_unfair_nodes=ratio_unfair_nodes,
        opposite_direction=opposite_direction,
        ratio_unfairness=ratio_unfairness,
        group_to_reduce=group_to_reduce,
        group_to_increment=group_to_increment,
        number_of_samples_per_node=number_of_samples_per_node,
        opposite_group_to_reduce=opposite_group_to_reduce,
        opposite_group_to_increment=opposite_group_to_increment,
        opposite_ratio_unfairness=opposite_ratio_unfairness,
        one_group_nodes=one_group_nodes,
    )
    disparities = Utils.compute_disparities_debug(client_data)
    Utils.plot_bar_plot(
        title=f"{approach}",
        disparities=disparities,
        nodes=[f"{i}" for i in range(len(client_data))],
    )
    print(disparities)

    return client_data, disparities, metadata  # , N_is, props_positive


def egalitarian_approach(x_data, y, z, num_nodes, number_of_samples_per_node=None):
    """
    With this approach we want to distribute the data among the nodes in an egalitarian way.
    This means that each node has the same amount of data and the same ratio of each group

    params:
    x_data: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
        len(y)//num_nodes
    """
    combinations = [(target, sensitive_value) for target, sensitive_value in zip(y, z, strict=False)]
    possible_combinations = set(combinations)
    data = {}
    for combination, x_, y_, z_ in zip(combinations, x_data, y, z, strict=False):
        if combination not in data:
            data[combination] = []
        data[combination].append({"x": x_, "y": y_, "z": z_})

    samples_from_each_group = min(list(Counter(combinations).values())) // num_nodes

    if number_of_samples_per_node:
        if samples_from_each_group * len(possible_combinations) < number_of_samples_per_node:
            msg = "Too many samples per node, choose a different number of samples per node"
            raise ValueError(msg)
        to_be_removed = (samples_from_each_group * len(possible_combinations) - number_of_samples_per_node) // len(
            possible_combinations
        )
        samples_from_each_group -= to_be_removed

    # create the nodes
    nodes = []
    for i in range(num_nodes):
        nodes.append([])
        # fill the nodes
        for combination in data:
            nodes[i].extend(data[combination][:samples_from_each_group])
            data[combination] = data[combination][samples_from_each_group:]

    return nodes, data


def _process_single_unfair_node(node, group_to_reduce, ratio_unfairness):
    node_data = []
    count_sensitive_group_samples = 0
    for sample in node:
        if (sample["y"], sample["z"]) == group_to_reduce:
            count_sensitive_group_samples += 1

    rng = np.random.default_rng()
    current_ratio = rng.uniform(ratio_unfairness[0], ratio_unfairness[1])
    samples_to_be_removed = int(count_sensitive_group_samples * current_ratio)
    samples_to_add = samples_to_be_removed

    removed_samples = []
    for sample in node:
        if (sample["y"], sample["z"]) == group_to_reduce and samples_to_be_removed > 0:
            samples_to_be_removed -= 1
            removed_samples.append(sample)
        else:
            node_data.append(sample)
    return node_data, removed_samples, samples_to_add


def create_unfair_nodes(
    fair_nodes: list,
    nodes_to_unfair: list,
    remaining_data: dict,
    group_to_reduce: tuple,
    group_to_increment: tuple,
    ratio_unfairness: tuple,
):
    """
    This function creates the unfair nodes.
    """
    unfair_nodes = []
    number_of_samples_to_add = []
    all_removed_samples = []

    for node in nodes_to_unfair:
        node_data, removed, to_add = _process_single_unfair_node(node, group_to_reduce, ratio_unfairness)
        unfair_nodes.append(node_data)
        all_removed_samples.extend(removed)
        number_of_samples_to_add.append(to_add)

    # Now we have to distribute the removed samples among the fair nodes
    if fair_nodes:
        max_samples_to_add = len(all_removed_samples) // len(fair_nodes)
        for node in fair_nodes:
            node.extend(all_removed_samples[:max_samples_to_add])
            all_removed_samples = all_removed_samples[max_samples_to_add:]

    if group_to_increment:
        _increment_unfair_groups(fair_nodes, unfair_nodes, remaining_data, group_to_increment, number_of_samples_to_add)

    return fair_nodes, unfair_nodes


def _increment_unfair_groups(fair_nodes, unfair_nodes, remaining_data, group_to_increment, number_of_samples_to_add):
    total_to_add = sum(number_of_samples_to_add)
    for node in fair_nodes:
        samples_to_remove = total_to_add // len(fair_nodes)
        for index, sample in enumerate(node):
            if (sample["y"], sample["z"]) == group_to_increment and samples_to_remove > 0:
                if group_to_increment not in remaining_data:
                    remaining_data[group_to_increment] = []
                remaining_data[group_to_increment].append(sample)
                samples_to_remove -= 1
                node.pop(index)
        if total_to_add > 0 and samples_to_remove != 0:
            msg = "Not enough samples to remove"
            raise ValueError(msg)

    if total_to_add > len(remaining_data.get(group_to_increment, [])):
        msg = "Too many samples to add"
        raise ValueError(msg)

    for node, samples_to_add in zip(unfair_nodes, number_of_samples_to_add, strict=False):
        node.extend(remaining_data[group_to_increment][:samples_to_add])
        remaining_data[group_to_increment] = remaining_data[group_to_increment][samples_to_add:]


def representative_diversity_approach(x_data, y, z, num_nodes, number_of_samples_per_node):
    """
    With this approach we want to distribute the data among the nodes in a representative diversity way.
    This means that each node has the same ratio of each group that we are observing in the dataset

    params:
    x_data: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
        len(y)//num_nodes
    """
    samples_per_node = number_of_samples_per_node if number_of_samples_per_node else len(y) // num_nodes
    # create the nodes sampling from the dataset wihout replacement
    dataset = [{"x": x_, "y": y_, "z": z_} for x_, y_, z_ in zip(x_data, y, z, strict=False)]
    # shuffle the dataset
    rng = np.random.default_rng()
    rng.shuffle(dataset)

    # Distribute the data among the nodes with a random sample from the dataset
    # considering the number of samples per node
    nodes = []
    for i in range(num_nodes):
        nodes.append([])
        nodes[i].extend(dataset[:samples_per_node])
        dataset = dataset[samples_per_node:]

    # Create the dictionary with the remaining data
    remaining_data = {}
    for sample in dataset:
        if (sample["y"], sample["z"]) not in remaining_data:
            remaining_data[(sample["y"], sample["z"])] = []
        remaining_data[(sample["y"], sample["z"])].append(sample)

    return nodes, remaining_data


def generate_clients_biased_data_mod(
    x_data,
    y,
    z,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    group_to_reduce: tuple | None = None,
    group_to_increment: tuple | None = None,
    number_of_samples_per_node: int | None = None,
    opposite_group_to_reduce: tuple | None = None,
    opposite_group_to_increment: tuple | None = None,
    opposite_ratio_unfairness: tuple | None = None,
    one_group_nodes: bool = False,
):
    """
    This function generates the data for the clients.

    params:
    x_data: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    approach: type of approach we want to use to distribute the data among the fair clients. This can be egalitarian or representative
    ratio_unfair_nodes: the fraction of unfair clients we want to have in the experiment
    opposite_direction: true if we want to allow different nodes to have different majoritiarian classes. For instance,
        we could have some nodes with a max disparity that depends on the majority class being 0 and other nodes with a max disparity
        that depends on the majority class being 1.
    group_to_reduce: the group that we want to be unfair. For instance, in the case of binary target and binary sensitive value
        we could have (0,0), (0,1), (1,0) or (1,1)
    ratio_unfairness: tuple (min, max) where min is the minimum ratio of samples that we want to remove from the group_to_reduce
        and max is the maximum ratio of samples that we want to remove from the group_to_reduce
    """
    # check if the number of samples that we want in each node is
    # greater than the number of samples we have in the dataset
    if number_of_samples_per_node and number_of_samples_per_node >= len(y) // num_nodes:
            msg = "Too many samples per node"
            raise ValueError(msg)    # check if the ratio_fair_nodes is between 0 and 1
    if ratio_unfair_nodes > 1:
        msg = "ratio_unfair_nodes must be less or equal than 1"
        raise ValueError(msg)
    if ratio_unfair_nodes < 0:
        msg = "ratio_unfair_nodes must be greater or equal than 0"
        raise ValueError(msg)
    if not group_to_reduce:
        msg = "group_to_reduce must be specified"
        raise ValueError(msg)
    if not group_to_increment:
        msg = "group_to_increment must be specified"
        raise ValueError(msg)
    # check if the approach type is egalitarian or representative
    if approach not in [
        "egalitarian",
        "representative",
    ]:
        msg = "Approach must be egalitarian or representative"
        raise ValueError(msg)

    number_unfair_nodes = int(num_nodes * ratio_unfair_nodes)
    number_fair_nodes = num_nodes - number_unfair_nodes
    if approach == "egalitarian":
        # first split the data among the nodes in an egalitarian way
        # each node has the same amount of data and the same ratio of each group
        nodes, remaining_data = egalitarian_approach(x_data, y, z, num_nodes, number_of_samples_per_node)
    else:
        nodes, remaining_data = representative_diversity_approach(x_data, y, z, num_nodes, number_of_samples_per_node)

    if opposite_direction:
        if not opposite_group_to_reduce:
            msg = "opposite_group_to_reduce must be specified"
            raise ValueError(msg)
        if not opposite_group_to_increment:
            msg = "opposite_group_to_increment must be specified"
            raise ValueError(msg)
        group_size = number_unfair_nodes // 2
        unfair_nodes_direction_1 = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes : number_fair_nodes + group_size],
            remaining_data=remaining_data,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            ratio_unfairness=ratio_unfairness,
        )
        unfair_nodes_direction_2 = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes + group_size :],
            remaining_data=remaining_data,
            group_to_reduce=opposite_group_to_reduce,
            group_to_increment=opposite_group_to_increment,
            ratio_unfairness=opposite_ratio_unfairness,
        )
        return (nodes[0:number_fair_nodes] + unfair_nodes_direction_1 + unfair_nodes_direction_2), [
            0
        ] * number_fair_nodes + [1] * len(unfair_nodes_direction_1)
    # At the moment this is the only thing that is working, we need
    # to fix the opposite direction version
    fair_nodes, unfair_nodes = create_unfair_nodes(
        fair_nodes=nodes[:number_fair_nodes],
        nodes_to_unfair=nodes[number_fair_nodes:],
        remaining_data=remaining_data,
        group_to_reduce=group_to_reduce,
        group_to_increment=group_to_increment,
        ratio_unfairness=ratio_unfairness,
    )

    if one_group_nodes:
        # create the nodes that only have one group
        fair_nodes, unfair_nodes = create_one_group_nodes(fair_nodes, unfair_nodes, ratio_unfair_nodes)
    return (
        fair_nodes + unfair_nodes,
        [0] * number_fair_nodes + [1] * number_fair_nodes,
    )


def create_one_group_nodes(fair_nodes, unfair_nodes, _ratio_unfair_nodes):
    num_one_group_nodes_fair = len(fair_nodes)  # int(num_one_group_nodes * (1 - ratio_unfair_nodes))
    len(unfair_nodes)  # num_one_group_nodes - num_one_group_nodes_fair

    removed_samples = {"0": [], "1": []}
    number_removed_samples = {}

    # Remove samples from the fair nodes and from the unfair nodes

    tmp_fair_nodes = []
    for node_id, node in enumerate(fair_nodes[:num_one_group_nodes_fair]):
        tmp_removed_samples = []
        tmp_samples = []
        for sample in node:
            if sample["z"] == 1 and node_id % 2 == 0:
                tmp_removed_samples.append(sample)
            else:
                tmp_samples.append(sample)

        tmp_fair_nodes.append(tmp_samples)
        removed_samples[str(node_id % 2)].extend(tmp_removed_samples)
        number_removed_samples[node_id] = len(tmp_removed_samples)

    return tmp_fair_nodes, unfair_nodes


def load_dutch(dataset_path):
    data = arff.loadarff(dataset_path + "dutch_census.arff")
    dutch_df = pd.DataFrame(data[0]).astype("int32")

    OCCUPATION_THRESHOLD = 300
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= OCCUPATION_THRESHOLD, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    dutch_df_feature_columns = [
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
        "sex_binary",
    ]

    metadata_dutch = {
        "name": "Dutch census",
        "code": ["DU1"],
        "protected_atts": ["sex_binary"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "occupation_binary",
    }

    return dutch_df, dutch_df_feature_columns, metadata_dutch


## Use this function to retrieve X, X, y arrays for training ML models
def dataset_to_numpy(
    _df,
    _feature_cols: list,
    _metadata: dict,
    num_sensitive_features: int = 1,
    sensitive_features_last: bool = True,
):
    """
    Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns

    """
    # transform features to 1-hot
    print(_feature_cols)
    print(_df.columns)
    _X = _df[_feature_cols]
    # take sensitive features separately
    print(f"Using {_metadata['protected_atts'][:num_sensitive_features]} as sensitive feature(s).")
    num_sensitive_features = min(num_sensitive_features, len(_metadata["protected_atts"]))
    _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
    _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])

    # 1-hot encode and scale features
    dummy_cols = _metadata.get("dummy_cols")
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    # original
    BINARY_CARDINALITY = 2
    # current implementation assumes each sensitive feature is binary
    for _i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        if len(_Z[tmp].unique()) != BINARY_CARDINALITY:
            msg = "Sensitive feature is not binary!"
            raise ValueError(msg)

    # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
    _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
    if sensitive_features_last:
        for i, tmp in enumerate(_Z.columns):
            if _metadata["protected_att_values"][i] not in _Z[tmp].unique():
                msg = "Protected attribute value not found in data!"
                raise ValueError(msg)
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
    # change booleans to floats

    # original
    _Z = _Z2.to_numpy()

    _y = _df[_metadata["target_variable"]].to_numpy()
    return _X, _Z, _y


# Use this function to retrieve X, X, y arrays for training ML models
def dataset_to_numpy_mod(
    _df,
    _feature_cols: list,
    _metadata: dict,
    num_sensitive_features: int = 1,
    sensitive_features_last: bool = True,
):
    """
    Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns

    """
    # transform features to 1-hot
    print(_feature_cols)

    _X = _df[_feature_cols]
    # take sensitive features separately
    print(f"Using {_metadata['protected_atts'][:num_sensitive_features]} as sensitive feature(s).")
    num_sensitive_features = min(num_sensitive_features, len(_metadata["protected_atts"]))
    _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]

    my_sensitive_features = _X[["edu_level"]]

    # 1-hot encode and scale features
    dummy_cols = _metadata.get("dummy_cols")
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    # original
    BINARY_CARDINALITY = 2
    # current implementation assumes each sensitive feature is binary
    for _i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        if len(_Z[tmp].unique()) != BINARY_CARDINALITY:
            msg = "Sensitive feature is not binary!"
            raise ValueError(msg)

    # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
    _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
    if sensitive_features_last:
        for i, tmp in enumerate(_Z.columns):
            if _metadata["protected_att_values"][i] not in _Z[tmp].unique():
                msg = "Protected attribute value not found in data!"
                raise ValueError(msg)
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
    # change booleans to floats

    # original
    _Z = _Z2.to_numpy()

    _Z = my_sensitive_features.to_numpy()
    print(_Z)

    _y = _df[_metadata["target_variable"]].to_numpy()
    return _X, _Z, _y


def get_tabular_numpy_dataset(dataset_name, num_sensitive_features, dataset_path=None):
    if dataset_name == "dutch":
        tmp = load_dutch(dataset_path=dataset_path)
    else:
        msg = "Unknown dataset name!"
        raise ValueError(msg)
    _X, _Z, _y = dataset_to_numpy(*tmp, num_sensitive_features=num_sensitive_features)
    return _X, _Z, _y


def _prepare_income_like_data(dataset_path, splitted_data_dir, num_nodes, cross_silo, sweep, validation_seed, seed):
    for client_name in range(num_nodes):
        client_dir = Path(dataset_path) / splitted_data_dir / str(client_name)
        train_pt = client_dir / "train.pt"
        if train_pt.exists():
            train_pt.unlink()

        X_train = np.load(client_dir / f"income_dataframes_{client_name}_train.npy", allow_pickle=True)
        Y_train = np.load(client_dir / f"income_labels_{client_name}_train.npy", allow_pickle=True)
        Z_train = np.load(client_dir / f"income_groups_{client_name}_train.npy", allow_pickle=True)
        W_train = np.load(client_dir / f"income_second_groups_{client_name}_train.npy", allow_pickle=True)
        T_train = np.load(client_dir / f"income_third_groups_{client_name}_train.npy", allow_pickle=True)

        if cross_silo:
            X_test = np.load(client_dir / f"income_dataframes_{client_name}_test.npy", allow_pickle=True)
            Y_test = np.load(client_dir / f"income_labels_{client_name}_test.npy", allow_pickle=True)
            Z_test = np.load(client_dir / f"income_groups_{client_name}_test.npy", allow_pickle=True)
            W_test = np.load(client_dir / f"income_second_groups_{client_name}_test.npy", allow_pickle=True)
            T_test = np.load(client_dir / f"income_third_groups_{client_name}_test.npy", allow_pickle=True)

            if sweep:
                (X_train, X_val, Y_train, Y_val, Z_train, Z_val, W_train, W_val, T_train, T_val) = train_test_split(
                    X_train, Y_train, Z_train, W_train, T_train, test_size=0.2, random_state=validation_seed
                )
                val_dataset = TabularDataset(
                    x=np.hstack((X_val, np.ones((X_val.shape[0], 1)))).astype(np.float32),
                    z=[item.item() for item in Z_val],
                    w=[item.item() for item in W_val],
                    t=[item.item() for item in T_val],
                    y=[item.item() for item in Y_val],
                )
                torch.save(val_dataset, client_dir / "val.pt")

            train_dataset = TabularDataset(
                x=np.hstack((X_train, np.ones((X_train.shape[0], 1)))).astype(np.float32),
                z=[item.item() for item in Z_train],
                w=[item.item() for item in W_train],
                t=[item.item() for item in T_train],
                y=[item.item() for item in Y_train],
            )
            random.seed(validation_seed)
            train_dataset.shuffle(seed=validation_seed)
            random.seed(seed)
            torch.save(train_dataset, client_dir / "train.pt")

            test_dataset = TabularDataset(
                x=np.hstack((X_test, np.ones((X_test.shape[0], 1)))).astype(np.float32),
                z=[item.item() for item in Z_test],
                w=[item.item() for item in W_test],
                t=[item.item() for item in T_test],
                y=[item.item() for item in Y_test],
            )
            torch.save(test_dataset, client_dir / "test.pt")
        else:
            if sweep:
                (X_train, X_val, Y_train, Y_val, Z_train, Z_val, W_train, W_val, T_train, T_val) = train_test_split(
                    X_train, Y_train, Z_train, W_train, T_train, test_size=0.2, random_state=validation_seed
                )
                val_dataset = TabularDataset(
                    x=np.hstack((X_val, np.ones((X_val.shape[0], 1)))).astype(np.float32),
                    z=[item.item() for item in Z_val],
                    w=[item.item() for item in W_val],
                    t=[item.item() for item in T_val],
                    y=[item.item() for item in Y_val],
                )
                torch.save(val_dataset, client_dir / "val.pt")

            train_dataset = TabularDataset(
                x=np.hstack((X_train, np.ones((X_train.shape[0], 1)))).astype(np.float32),
                z=[item.item() for item in Z_train],
                w=[item.item() for item in W_train],
                t=[item.item() for item in T_train],
                y=[item.item() for item in Y_train],
            )
            random.seed(validation_seed)
            train_dataset.shuffle(seed=validation_seed)
            random.seed(seed)
            torch.save(train_dataset, client_dir / "train.pt")

    return f"{dataset_path}/{splitted_data_dir}"


def _prepare_dutch_prepared_data(dataset_path, splitted_data_dir, num_nodes, cross_silo, sweep, validation_seed, seed):
    for client_name in range(num_nodes):
        if cross_silo:
            path_train = Path(dataset_path) / f"train_train_{client_name}.csv"
            path_test = Path(dataset_path) / f"train_test_{client_name}.csv"
            if not path_train.exists() or not path_test.exists():
                continue
            df_train, df_test = pd.read_csv(path_train), pd.read_csv(path_test)
            len_train = len(df_train)
            dutch_df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            path_train = Path(dataset_path) / f"train_{client_name}.csv"
            if not path_train.exists():
                continue
            dutch_df = pd.read_csv(path_train)
            len_train = len(dutch_df)

        dutch_df = dutch_df.astype("int32")
        for col in ["sex", "occupation"]:
            if col in dutch_df.columns:
                del dutch_df[col]

        feature_cols = [
            "age", "household_position", "household_size", "prev_residence_place",
            "citizenship", "country_birth", "edu_level", "economic_status",
            "cur_eco_activity", "Marital_status", "sex_binary"
        ]
        metadata = {
            "name": "Dutch census", "code": ["DU1"], "protected_atts": ["sex_binary"],
            "protected_att_values": [0], "protected_att_descriptions": ["Gender = Female"],
            "target_variable": "occupation_binary",
        }

        z_full = dutch_df["sex_binary"].to_numpy().astype(np.float32)
        w_full = dutch_df["Marital_status"].to_numpy().astype(np.float32)
        x_full, _, y_full = dataset_to_numpy(dutch_df, feature_cols, metadata, num_sensitive_features=1)

        x_train_raw = x_full[:len_train]
        z_train_raw = z_full[:len_train]
        y_train_raw = y_full[:len_train]
        w_train_raw = w_full[:len_train]

        client_dir = Path(dataset_path) / splitted_data_dir / str(client_name)
        client_dir.mkdir(parents=True, exist_ok=True)
        for f in client_dir.glob("*.pt"):
            f.unlink()

        if sweep:
            X_train, X_val, Y_train, Y_val, Z_train, Z_val, W_train, W_val = train_test_split(
                x_train_raw, y_train_raw, z_train_raw, w_train_raw, test_size=0.2, random_state=validation_seed
            )
            val_ds = TabularDataset(x=np.hstack((X_val, np.ones((X_val.shape[0], 1)))).astype(np.float32), z=Z_val, y=Y_val, w=W_val)
            torch.save(val_ds, client_dir / "val.pt")
        else:
            X_train, Y_train, Z_train, W_train = x_train_raw, y_train_raw, z_train_raw, w_train_raw

        train_ds = TabularDataset(x=np.hstack((X_train, np.ones((X_train.shape[0], 1)))).astype(np.float32), z=Z_train, y=Y_train, w=W_train)
        random.seed(seed)
        train_ds.shuffle(seed=seed)
        torch.save(train_ds, client_dir / "train.pt")

        if cross_silo:
            test_ds = TabularDataset(
                x=np.hstack((x_full[len_train:], np.ones((x_full[len_train:].shape[0], 1)))).astype(np.float32),
                z=z_full[len_train:], y=y_full[len_train:], w=w_full[len_train:]
            )
            torch.save(test_ds, client_dir / "test.pt")

    return f"{dataset_path}/{splitted_data_dir}"


def _prepare_celeba_prepared_data(dataset_path, splitted_data_dir, num_nodes, cross_silo, sweep, validation_seed):
    clean_path = Path(dataset_path.rstrip("/"))
    celeba_root = clean_path.parent.parent
    json_path = celeba_root / "celeba_img_dict.json"
    img_map = {}
    if json_path.exists():
        with json_path.open() as f:
            img_map = json.load(f)

    for client_name in range(num_nodes):
        client_dir = Path(dataset_path) / splitted_data_dir / str(client_name)
        client_dir.mkdir(parents=True, exist_ok=True)
        for f in client_dir.glob("*.pt"):
            f.unlink()

        transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        splits = [("train", f"{dataset_path}/train_train_{client_name}.csv"), ("test", f"{dataset_path}/train_test_{client_name}.csv")] if cross_silo else [("train", f"{dataset_path}/train_{client_name}.csv")]

        processed_data = {}
        for split_name, csv_path_str in splits:
            csv_path = Path(csv_path_str)
            if not csv_path.exists():
                processed_data[split_name] = None
                continue
            df = pd.read_csv(csv_path)
            labels = df["Smiling"].tolist()
            male_attr = df["Male"].tolist()
            hair_attr = df["hair_color"].tolist() if "hair_color" in df.columns else [0] * len(labels)
            is_val_exp = "value" in dataset_path.lower()
            z_attr = hair_attr if is_val_exp else male_attr
            w_attr = male_attr if is_val_exp else hair_attr
            img_ids = df["celeb_id"].tolist() if "celeb_id" in df.columns else []
            processed_data[split_name] = {"image_ids": img_ids, "labels": labels, "sensitive": z_attr, "second_sensitive": w_attr}

        if processed_data.get("train"):
            t_data = processed_data["train"]
            if sweep:
                X_tr, X_va, y_tr, y_va, z_tr, z_va, w_tr, w_va = train_test_split(
                    t_data["image_ids"], t_data["labels"], t_data["sensitive"], t_data["second_sensitive"],
                    test_size=0.2, random_state=validation_seed
                )
                torch.save(CelebaPreparedDataset(X_va, img_map, y_va, z_va, w_va, transform), client_dir / "val.pt")
                t_data.update({"image_ids": X_tr, "labels": y_tr, "sensitive": z_tr, "second_sensitive": w_tr})
            torch.save(CelebaPreparedDataset(t_data["image_ids"], img_map, t_data["labels"], t_data["sensitive"], t_data["second_sensitive"], transform), client_dir / "train.pt")

        if processed_data.get("test"):
            t_data = processed_data["test"]
            torch.save(CelebaPreparedDataset(t_data["image_ids"], img_map, t_data["labels"], t_data["sensitive"], t_data["second_sensitive"], transform), client_dir / "test.pt")

    return f"{dataset_path}/{splitted_data_dir}"


def _save_fed_metadata(data_dir, possible_z, possible_y, client_data):
    possible_y_str = [str(int(item)) for item in possible_y.tolist()]
    possible_z_str = [str(int(item)) for item in possible_z.tolist()]
    all_combinations, missing_combinations, sent_disparity_combinations = [], [], [f"1|{s}" for s in possible_z_str]
    for comb in sent_disparity_combinations:
        missing_combinations.append(("0" + comb[1:], comb))
        all_combinations.extend([comb, "0" + comb[1:]])

    with (data_dir / "metadata.json").open("w") as f:
        json.dump({"possible_z": possible_z_str, "possible_y": possible_y_str, "missing_combinations": missing_combinations, "all_combinations": all_combinations, "combinations": sent_disparity_combinations}, f, indent=4)

    preds = [[int(y) for y in c["y"]] for c in client_data]
    sfs = [[int(z) for z in c["z"]] for c in client_data]
    Utils.plot_distributions("Distribution of the nodes", Utils.compute_distribution_debug(preds, sfs), [f"{i}" for i in range(len(client_data))], all_combinations)


def prepare_tabular_data(
    dataset_path: str,
    dataset_name: str,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    group_to_reduce: tuple | None = None,
    group_to_increment: tuple | None = None,
    number_of_samples_per_node: int | None = None,
    opposite_group_to_reduce: tuple | None = None,
    opposite_group_to_increment: tuple | None = None,
    opposite_ratio_unfairness: tuple | None = None,
    one_group_nodes: bool = False,
    splitted_data_dir: str | None = None,
    cross_silo: bool = False,
    sweep: bool = False,
    seed: int = 42,
    validation_seed: int = 42,
):
    if dataset_name in {"income", "employment", "employment_NO_RACE", "income_NO_RACE", "income_cross_device"}:
        return _prepare_income_like_data(dataset_path, splitted_data_dir, num_nodes, cross_silo, sweep, validation_seed, seed), None

    if dataset_name == "dutch_prepared":
        return _prepare_dutch_prepared_data(dataset_path, splitted_data_dir, num_nodes, cross_silo, sweep, validation_seed, seed), None

    if dataset_name == "celeba_prepared":
        return _prepare_celeba_prepared_data(dataset_path, splitted_data_dir, num_nodes, cross_silo, sweep, validation_seed), None

    client_data, disparities, metadata = get_tabular_data(
        dataset_name=dataset_name,
        num_sensitive_features=1,
        dataset_path=dataset_path,
        approach=approach,
        num_nodes=num_nodes,
        ratio_unfair_nodes=ratio_unfair_nodes,
        opposite_direction=opposite_direction,
        ratio_unfairness=ratio_unfairness,
        group_to_reduce=group_to_reduce,
        group_to_increment=group_to_increment,
        number_of_samples_per_node=number_of_samples_per_node,
        opposite_group_to_reduce=opposite_group_to_reduce,
        opposite_group_to_increment=opposite_group_to_increment,
        opposite_ratio_unfairness=opposite_ratio_unfairness,
        one_group_nodes=one_group_nodes,
    )

    client_data_formatted = []
    possible_z, possible_y = np.array([]), np.array([])
    for client in client_data:
        tmp_x, tmp_y, tmp_z = [], [], []
        for sample in client:
            tmp_x.append(sample["x"])
            tmp_y.append(sample["y"])
            tmp_z.append(sample["z"])
        client_data_formatted.append({"x": np.array(tmp_x), "y": np.array(tmp_y), "z": np.array(tmp_z)})
        possible_z = np.unique(np.concatenate((possible_z, np.unique(tmp_z))))
        possible_y = np.unique(np.concatenate((possible_y, np.unique(tmp_y))))

    data_dir = Path(dataset_path) / splitted_data_dir
    if data_dir.exists():
        for f in data_dir.glob("*"):
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

    for client_name, (client, client_disparity, client_metadata) in enumerate(
        zip(client_data_formatted, disparities, metadata, strict=False)
    ):
        client_dir = data_dir / str(client_name)
        client_dir.mkdir(parents=True, exist_ok=True)
        custom_dataset = TabularDataset(x=np.hstack((client["x"], np.ones((client["x"].shape[0], 1)))).astype(np.float32), z=client["z"], y=client["y"])
        torch.save(custom_dataset, client_dir / "train.pt")
        with (client_dir / "metadata.json").open("w") as outfile:
            json.dump(Utils.get_dataset_statistics(custom_dataset, client_disparity, client_metadata), outfile, indent=4)

    _save_fed_metadata(data_dir, possible_z, possible_y, client_data_formatted)
    return str(data_dir), client_data_formatted
