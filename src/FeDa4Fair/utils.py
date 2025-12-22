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

"""Helper functions."""

from typing import Any

import pandas as pd


def drop_data(
    df: pd.DataFrame,
    percentage: float,
    column1: str,
    value1: Any,
    label_column: str,
    column2: str | None = None,
    value2: Any | None = None,
) -> pd.DataFrame:
    """
    Drop a percentage of rows from a DataFrame that match specific criteria.

    This function removes a given percentage of rows with label_name=True where the value in `column1` matches `value1`
    and optionally, where `column2` matches `value2`.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be filtered.

    percentage : float
        A float between 0 and 1 representing the fraction of matching rows to drop.

    column1 : str
        The name of the first column to filter by.

    value1 : Any
        The value in `column1` that must match for a row to be considered for dropping.

    label_column : str
        The name of the label_name column. Label values are expected to be binary.

    column2 : Optional[str], default=None
        An optional second column to filter by.

    value2 : Optional[Any], default=None
        The value in `column2` that must also match (in conjunction with `column1`) for a
        row to be considered for dropping. Only used if `column2` is provided.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified percentage of matching rows removed.

    """
    if not (0 <= percentage <= 1):
        msg = "Fraction must be between 0 and 1"
        raise ValueError(msg)

    condition = df[column1] == value1
    if column2 is not None and value2 is not None:
        condition &= df[column2] == value2

    matching_rows = df[condition & (df[label_column])]
    num_to_drop = int(len(matching_rows) * percentage)
    rows_to_drop = matching_rows.sample(n=num_to_drop, random_state=42).index

    return df.drop(index=rows_to_drop)


def flip_data(
    df: pd.DataFrame,
    percentage: float,
    column1: str,
    value1: Any,
    label_column: str,
    column2: str | None = None,
    value2: Any | None = None,
) -> pd.DataFrame:
    """
    Flip the label_name from True to False of a percentage of rows matching specified criteria.

    This function modifies the DataFrame by flipping the value in the `label_column` from True to False
     for a specified percentage of rows where `column1 == value1`.
    Optionally, the flip is further constrained to rows where `column2 == value2`.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame whose labels will be modified.

    percentage : float
        A float between 0 and 1 representing the fraction of matching rows whose label_name should be flipped.

    column1 : str
        The name of the first column used to filter the rows to be considered for flipping.

    value1 : Any
        The value in `column1` that must match for a row to be eligible for label_name flipping.

    label_column : str
        The name of the column containing the labels to be flipped. Label values are expected to be binary.

    column2 : Optional[str], default=None
        An optional second column used to refine the filtering condition.

    value2 : Optional[Any], default=None
        The value in `column2` that must also match (in conjunction with `column1`) for a
        row to be eligible for label_name flipping. Only used if `column2` is provided.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified percentage of labels flipped in the filtered subset.

    """
    if not (0 <= percentage <= 1):
        msg = "Fraction must be between 0 and 1"
        raise ValueError(msg)

    condition = df[column1] == value1
    if column2 is not None and value2 is not None:
        condition &= df[column2] == value2

    matching_rows = df[condition & (df[label_column])]
    num_to_flip = int(len(matching_rows) * percentage)

    if num_to_flip == 0:
        return df

    rows_to_flip = matching_rows.sample(n=num_to_flip, random_state=42).index
    df.loc[rows_to_flip, label_column] = False

    return df
