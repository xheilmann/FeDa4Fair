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
    sens_att: str,
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
    sens_att : str
        Name of the sensitive attribute column.
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

    diff_df = pd.Series(diff_matrix.flatten(), index=column_names)

    if size_unit == "value":
        # Return max diff and the pair responsible
        return pd.Series(
            [diff_df.max(), diff_df.idxmax()],
            index=[f"{sens_att}_{fairness_metric}", f"{sens_att}_val"],
        )
    if size_unit == "attribute":
        # Return only the max difference
        return pd.Series(
            [diff_df.max(), diff_df.max()],
            index=[f"{sens_att}_{fairness_metric}", f"{sens_att}_val"],
        )

    # "attribute-value" returns all pairwise differences
    return diff_df


def compute_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    model: Any,
    sens_att: str,
    max_num_partitions: int | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    label_name: str = "label",
    sens_cols: list[str] | None = None,
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
) -> pd.DataFrame:
    """
    Computes fairness metrics across dataset partitions.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner containing the training/reference data.
    partitioner_test : Partitioner
        Partitioner containing the test data for evaluation.
    model : Any
        Model to evaluate. If provided, it is trained on `partitioner` data and evaluated on `partitioner_test`.
        If None, data labels are used directly (data bias check).
    sens_att : str
        Name of the sensitive attribute column.
    max_num_partitions : Optional[int], default=None
        Limit the number of partitions to evaluate.
    fairness_metric : Literal["DP", "EO"], default="DP"
        Metric to compute.
    label_name : str, default="label"
        Name of the label column.
    sens_cols : Optional[list[str]], default=None
        List of sensitive attributes to drop from features before training/inference.
    size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
        Detail level of result.

    Returns
    -------
    pd.DataFrame
        Fairness metrics for each partition.

    """
    if sens_cols is None:
        sens_cols = []

    if max_num_partitions is None:
        num_parts = partitioner.num_partitions
    else:
        num_parts = min(max_num_partitions, partitioner.num_partitions)

    partition_id_to_fairness = {}

    for partition_id in range(num_parts):
        partition = partitioner.load_partition(partition_id)
        partition_test_data = partitioner_test.load_partition(partition_id)

        if model is not None:
            # Training and Prediction Mode
            train_df = partition.to_pandas()
            test_df = partition_test_data.to_pandas()

            # Prepare Training Data
            cols_to_drop = [*sens_cols, label_name]
            x_train = train_df.drop(columns=cols_to_drop, errors="ignore")
            y_train = train_df[label_name].to_numpy().flatten()

            # Train Model
            model.fit(x_train, y_train)

            # Prepare Test Data
            x_test = test_df.drop(columns=cols_to_drop, errors="ignore")
            y_pred = model.predict(x_test)

            y_true = test_df[label_name].to_numpy()
            acc = accuracy_score(y_true, y_pred)

            sf_data = test_df[[sens_att]]  # Keep as DataFrame for fairlearn

        else:
            # Data Bias Mode (No Model)
            raw_df = partition.to_pandas()
            y_true = raw_df[[label_name]]
            y_pred = raw_df[[label_name]]  # "Prediction" is just the label
            sf_data = raw_df[[sens_att]]
            acc = None

        fairness_series = _compute_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sf_data=sf_data,
            fairness_metric=fairness_metric,
            sens_att=sens_att,
            size_unit=size_unit,
        )

        if acc is not None:
            fairness_series["Accuracy"] = acc

        partition_id_to_fairness[partition_id] = fairness_series

    dataframe = pd.DataFrame.from_dict(partition_id_to_fairness, orient="index")
    dataframe.index.name = "Partition ID"

    return dataframe
