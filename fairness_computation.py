from itertools import product
import numpy as np
from typing import Literal
from typing import  Optional
from flwr_datasets.partitioner import Partitioner
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 80)
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate




def _compute_fairness(y_true, y_pred, sf_data, fairness_metric, column_name, size_unit):
    if fairness_metric == "DP":

        sel_rate = MetricFrame(
            metrics={"sel":selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf_data,
            )
        df = sel_rate.by_group
        diff_matrix = df['sel'].values[:, None] - df['sel'].values[None, :]
        index = df.index.values
        column_names = [f"{index[i]}_{index[j]}" for i, j in product(range(len(df)), repeat=2)]


    elif fairness_metric == "EO":
        tpr = MetricFrame(
            metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf_data,
        )
        # Compute the difference matrices
        df = tpr.by_group
        diff_matrix_col2 = df['tpr'].values[:, None] - df['tpr'].values[None, :]
        diff_matrix_col3 = df['fpr'].values[:, None] - df['fpr'].values[None, :]

        # Compute the absolute differences
        abs_diff_col2 = np.abs(diff_matrix_col2)
        abs_diff_col3 = np.abs(diff_matrix_col3)

        # Find the mask where Column2's absolute difference is larger
        mask = abs_diff_col2 >= abs_diff_col3
        index = df.index.values
        column_names = [f"{index[i]}_{index[j]}" for i, j in
                        product(range(len(df)), repeat=2)]

        # Create the final matrix using the original differences
        diff_matrix = np.where(mask, diff_matrix_col2, diff_matrix_col3)

    else:
        raise ValueError(f"Unknown fairness metric {fairness_metric}")


    diff_df = pd.Series(diff_matrix.flatten(), index=column_names)

    if size_unit =="value":
        diff_df = pd.Series([diff_df.max(),diff_df.idxmax()] , index=[f"{column_name}_{fairness_metric}", f"{column_name}_val"])
    if size_unit =="attribute":
        diff_df = pd.Series([diff_df.max(),diff_df.max()] , index=[f"{column_name}_{fairness_metric}", f"{column_name}_val"])


    return diff_df


def compute_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    model: None,
    column_name: str,
    max_num_partitions: Optional[int] = None,
    fairness_metric="DP",
    label: str = "label",
    sensitive_attributes:list[str]=[],
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
) -> pd.DataFrame:
    """Compute the counts of unique values in a given column in the partitions.

    Take into account all possible labels in dataset when computing count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying label based on which the count will be calculated.
    verbose_names : bool
        Whether to use verbose versions of the values in the column specified by
        `column_name`. The verbose values are possible to extract if the column is a
        feature of type `ClassLabel`.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    dataframe: pd.DataFrame
        DataFrame where the row index represent the partition id and the column index
        represent the unique values found in column specified by `column_name`
        (e.g. represeting the labels). The value of the dataframe.loc[i, j] represents
        the count of the label j, in the partition of index i.




    :param fairness_metric:
    :param label:
    """
    #if column_name not in partitioner.dataset.column_names:
    #    raise ValueError(
     #       f"The specified 'column_name': '{column_name}' is not present in the "
    #        f"dataset. The dataset contains columns {partitioner.dataset.column_names}."
     #   )
    if label is None or label not in partitioner.dataset.column_names:
        raise ValueError(f"The specified 'label' is not present in the dataset or was not set. ")
    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    assert isinstance(max_num_partitions, int)
    partition_id_to_fairness= {}
    for partition_id in range(max_num_partitions):
        partition = partitioner.load_partition(partition_id)
        partition_test =partitioner_test.load_partition(partition_id)

        if model is not None:
            train = partition.remove_columns(sensitive_attributes.append(label)).to_pandas()
            train_labels = partition.select_columns(label).to_pandas()
            model.fit(train, train_labels)
            y_pred = model.predict(partition_test.remove_columns(sensitive_attributes.append(label)).to_pandas())
            y_true = partition_test.select_columns(label).to_pandas()
            sf_data = partition_test.select_columns(column_name).to_pandas()

        else:
            y_true = partition.select_columns(label).to_pandas()
            y_pred = partition.select_columns(label).to_pandas()
            sf_data = partition.select_columns(column_name).to_pandas()
        partition_id_to_fairness[partition_id] = _compute_fairness(
        y_true=y_true,y_pred=y_pred,sf_data=sf_data, fairness_metric=fairness_metric, column_name=column_name, size_unit = size_unit)

    dataframe = pd.DataFrame.from_dict(
        partition_id_to_fairness, orient="index"
    )
    dataframe.index.name = "Partition ID"

    return dataframe


