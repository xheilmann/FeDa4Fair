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
    """ Compute a fairness metric (Demographic Parity or Equalized Odds) for given sensitive/s attribute/s specified in column name.

    This function supports group-based fairness metrics and allows different levels of evaluation
    depending on the `size_unit` parameter.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels for the predictions.

    y_pred : array-like
        Model predictions corresponding to `y_true`.

    sf_data : pd.DataFrame
        DataFrame containing the sensitive feature column specified by `column_name`.
        Must align in length and order with `y_true` and `y_pred`.

    fairness_metric : str
        Fairness metric to compute:
        - "DP": Demographic Parity
        - "EO": Equalized Odds

    column_name : str
        Name of the sensitive attribute used for fairness evaluation (e.g., "SEX", "RACE").

    size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
       The level at which fairness is evaluated:
        - "attribute": only worst fairness metric is returned,
        - "value": worst fairness metric as well as for which values this fairness was calculated is returned,
        - "attribute-value": all possible fairness metric values are returned.

    Returns
    -------
    dict
        A dictionary where keys reflect the evaluated fairness depending on size_unit.
    """
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
    """
    Computes the specified fairness metric for a given column or an intersection of columns across dataset partitions.

    Parameters
    ----------
    partitioner : Partitioner
        A `Partitioner` object containing the training dataset.

    partitioner_test : Partitioner
        A `Partitioner` object for the test set, used in fairness analysis if a model is specified.

    model : optional
        Model used for prediction-based fairness metrics like Equalized Odds (EO).

    column_name : str
        Name of the column which should be evaluated for fairness. If more than one then intersectional fairness is evaluated.

    max_num_partitions : Optional[int], default=None
        Maximum number of partitions to consider. If `None`, all partitions are included.

    fairness_metric : Literal["DP", "EO"], default="DP"
        Fairness metric to use for evaluation:
        - "DP": Demographic Parity
        - "EO": Equalized Odds

    label : str, default="label"
        Name of the label column, used particularly when evaluating fairness metrics.

    sensitive_attributes : list of str, default=[]
        List of sensitive attribute column names which are deleted before a model is trained on the dataset.

    size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
       The level at which fairness is evaluated:
        - "attribute": only worst fairness metric is returned,
        - "value": worst fairness metric as well as for which values this fairness was calculated is returned,
        - "attribute-value": all possible fairness metric values are returned.

    Returns
    -------
    pd.DataFrame
        A DataFrame where:
        - Rows represent partition IDs
        - Columns represent what is specified in size_unit
    """
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


