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

"""Functions to compute fairness metrics."""

from collections.abc import Callable
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from flwr_datasets.partitioner import Partitioner
from sklearn.metrics import accuracy_score

# Configure pandas display options
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 80)


def _compute_fairness(
    y_true: Any,
    y_pred: Any,
    sf_data: pd.DataFrame,
    fairness_metric: Literal["DP", "EO"],
    sens_att: str | list[str],
    size_unit: Literal["value", "attribute", "attribute-value"],
) -> pd.Series:
    """
    Compute a fairness metric (Demographic Parity or Equalized Odds) for given sensitive attribute(s).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Model predictions.
    sf_data : pd.DataFrame
        DataFrame containing sensitive feature(s).
    fairness_metric : Literal["DP", "EO"]
        "DP" for Demographic Parity, "EO" for Equalized Odds.
    sens_att : str | list[str]
        Name(s) of the sensitive attribute column(s).
    size_unit : Literal["value", "attribute", "attribute-value"]
        Level of detail for the returned metric.

    Returns
    -------
    pd.Series
        Series containing the computed fairness metric values.

    """
    if fairness_metric == "DP":
        # Demographic Parity: difference in selection rates
        sel_rate = MetricFrame(
            metrics={"sel": selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf_data,
        )
        group_df = sel_rate.by_group
        # Compute pairwise differences
        diff_matrix = group_df["sel"].to_numpy()[:, None] - group_df["sel"].to_numpy()[None, :]
        index = group_df.index.to_numpy()
        column_names = [f"{index[i]}_{index[j]}" for i, j in product(range(len(group_df)), repeat=2)]

    elif fairness_metric == "EO":
        # Equalized Odds: difference in TPR and FPR
        tpr = MetricFrame(
            metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf_data,
        )
        group_df = tpr.by_group

        # Calculate differences for TPR and FPR
        diff_matrix_tpr = group_df["tpr"].to_numpy()[:, None] - group_df["tpr"].to_numpy()[None, :]
        diff_matrix_fpr = group_df["fpr"].to_numpy()[:, None] - group_df["fpr"].to_numpy()[None, :]

        # We take the larger absolute difference (worst case)
        abs_diff_tpr = np.abs(diff_matrix_tpr)
        abs_diff_fpr = np.abs(diff_matrix_fpr)

        mask = abs_diff_tpr >= abs_diff_fpr
        diff_matrix = np.where(mask, diff_matrix_tpr, diff_matrix_fpr)

        index = group_df.index.to_numpy()
        column_names = [f"{index[i]}_{index[j]}" for i, j in product(range(len(group_df)), repeat=2)]

    else:
        msg = f"Unknown fairness metric {fairness_metric}"
        raise ValueError(msg)

    sens_att_name = str(sens_att) if isinstance(sens_att, list) else sens_att
    diff_df = pd.Series(diff_matrix.flatten(), index=[f"{sens_att_name}_{c}" for c in column_names])

    if size_unit == "value":
        # Return max diff and the pair responsible
        return pd.Series(
            [diff_df.max(), diff_df.idxmax()],
            index=[f"{sens_att_name}_{fairness_metric}", f"{sens_att_name}_val"],
        )
    if size_unit == "attribute":
        # Return only the max difference
        return pd.Series(
            [diff_df.max(), diff_df.max()],
            index=[f"{sens_att_name}_{fairness_metric}", f"{sens_att_name}_val"],
        )

    # "attribute-value" returns all pairwise differences
    return diff_df


def compute_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    model: Any,
    sens_att: str | list[str],
    max_num_partitions: int | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    label_name: str = "label",
    sens_cols: list[str] | None = None,
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    progress_callback: Callable[[int], None] | None = None,
    fds: Any | None = None,
    split: str | None = None,
    test_split: str | None = None,
) -> pd.DataFrame:
    """
    Computes fairness metrics across dataset partitions.
    """
    if sens_cols is None:
        sens_cols = []

    num_parts = min(max_num_partitions or float("inf"), partitioner.num_partitions)
    partition_id_to_fairness = {}

    for partition_id in range(int(num_parts)):
        if progress_callback is not None:
            progress_callback(partition_id)

        # If fds and split are provided, use fds.load_partition to include bias injection
        if fds is not None and split is not None:
            partition = fds.load_partition(partition_id, split=split)
            # Use test_split if provided, otherwise fallback to split
            actual_test_split = test_split if test_split is not None else split
            partition_test_data = fds.load_partition(partition_id, split=actual_test_split)
        else:
            partition = partitioner.load_partition(partition_id)
            partition_test_data = partitioner_test.load_partition(partition_id)

        if model is not None:
            fairness_series = _evaluate_model_on_partition(
                model, partition, partition_test_data, sens_att, fairness_metric, label_name, sens_cols, size_unit
            )
        else:
            fairness_series = _evaluate_data_bias_on_partition(
                partition, sens_att, fairness_metric, label_name, size_unit
            )

        partition_id_to_fairness[partition_id] = fairness_series
        partition_id_to_fairness[partition_id]["Sample Count"] = len(partition)

    dataframe = pd.DataFrame.from_dict(partition_id_to_fairness, orient="index")
    dataframe.index.name = "Partition ID"
    return dataframe


def compute_multi_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    model: Any,
    sens_atts: list[str],
    max_num_partitions: int | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    label_name: str = "label",
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    fds: Any | None = None,
    split: str | None = None,
    test_split: str | None = None,
) -> pd.DataFrame:
    """
    Computes fairness metrics for multiple sensitive attributes independently.
    Trains the model only once per partition.
    """
    num_parts = min(max_num_partitions or float("inf"), partitioner.num_partitions)
    partition_id_to_results = {}

    for partition_id in range(int(num_parts)):
        if fds is not None and split is not None:
            partition = fds.load_partition(partition_id, split=split)
            actual_test_split = test_split or split
            partition_test_data = fds.load_partition(partition_id, split=actual_test_split)
        else:
            partition = partitioner.load_partition(partition_id)
            partition_test_data = partitioner_test.load_partition(partition_id)

        train_df = partition.to_pandas()
        test_df = partition_test_data.to_pandas()

        if not (isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)):
            continue

        # Cleanup index if present
        if "__index_level_0__" in train_df.columns:
            train_df = train_df.drop(columns=["__index_level_0__"])
        if "__index_level_0__" in test_df.columns:
            test_df = test_df.drop(columns=["__index_level_0__"])

        res = {"Sample Count": len(train_df)}

        if model is not None:
            # Train model once
            cols_to_drop = [*sens_atts, label_name]
            x_train = train_df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include=["number", "bool"])
            y_train = train_df[label_name].to_numpy().flatten()

            x_test = test_df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include=["number", "bool"])
            y_test = test_df[label_name].to_numpy().flatten()

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            res["Accuracy"] = accuracy_score(y_test, y_pred)

            # Evaluate for each attribute
            for attr in sens_atts:
                sf_data = test_df[[attr]]
                fair_res = _compute_fairness(y_test, y_pred, sf_data, fairness_metric, attr, size_unit)
                res.update(fair_res.to_dict())
        else:
            # Data bias
            y_true = train_df[[label_name]]
            for attr in sens_atts:
                sf_data = train_df[[attr]]
                fair_res = _compute_fairness(y_true, y_true, sf_data, fairness_metric, attr, size_unit)
                res.update(fair_res.to_dict())

        partition_id_to_results[partition_id] = res

    dataframe = pd.DataFrame.from_dict(partition_id_to_results, orient="index")
    dataframe.index.name = "Partition ID"
    return dataframe


def _evaluate_model_on_partition(
    model, partition, partition_test_data, sens_att, fairness_metric, label_name, sens_cols, size_unit
):
    train_df, test_df = partition.to_pandas(), partition_test_data.to_pandas()
    if not (isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame)):
        msg = "Partition data is not a pandas DataFrame"
        raise TypeError(msg)

    # Cleanup index if present
    if "__index_level_0__" in train_df.columns:
        train_df = train_df.drop(columns=["__index_level_0__"])
    if "__index_level_0__" in test_df.columns:
        test_df = test_df.drop(columns=["__index_level_0__"])

    cols_to_drop = [*sens_cols, label_name]
    x_train, y_train = train_df.drop(columns=cols_to_drop, errors="ignore"), train_df[label_name].to_numpy().flatten()

    # Ensure features are numeric
    x_train = x_train.select_dtypes(include=["number", "bool"])

    model.fit(x_train, y_train)

    x_test = test_df.drop(columns=cols_to_drop, errors="ignore")
    x_test = x_test.select_dtypes(include=["number", "bool"])

    y_pred, y_true = model.predict(x_test), test_df[label_name].to_numpy()
    sf_data = test_df[sens_att] if isinstance(sens_att, list) else test_df[[sens_att]]

    fairness_series = _compute_fairness(y_true, y_pred, sf_data, fairness_metric, sens_att, size_unit)
    fairness_series["Accuracy"] = accuracy_score(y_true, y_pred)
    return fairness_series


def _evaluate_data_bias_on_partition(partition, sens_att, fairness_metric, label_name, size_unit):
    raw_df = partition.to_pandas()
    if not isinstance(raw_df, pd.DataFrame):
        msg = "Partition data is not a pandas DataFrame"
        raise TypeError(msg)

    # Cleanup index if present
    if "__index_level_0__" in raw_df.columns:
        raw_df = raw_df.drop(columns=["__index_level_0__"])

    y_true = raw_df[[label_name]]
    sf_data = raw_df[sens_att] if isinstance(sens_att, list) else raw_df[[sens_att]]
    return _compute_fairness(y_true, y_true, sf_data, fairness_metric, sens_att, size_unit)
