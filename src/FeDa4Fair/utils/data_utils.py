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

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


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
    if pd.api.types.is_bool_dtype(df[label_column]):
        df.loc[rows_to_flip, label_column] = False
    else:
        df.loc[rows_to_flip, label_column] = 0

    return df


def balance_data(
    df: pd.DataFrame,
    column1: str,
    label_column: str,
) -> tuple[pd.DataFrame, int]:
    """
    Balances both group sizes and selection rates across groups defined by column1.
    Ensures every group has the same number of positive and negative samples.

    Returns:
        tuple[pd.DataFrame, int]: The balanced DataFrame and the total number of samples removed.

    """
    groups = df[column1].unique()
    if len(groups) <= 1:
        return df, 0

    # Count positive and negative samples per group
    pos_counts = {}
    neg_counts = {}
    for group in groups:
        group_df = df[df[column1] == group]
        pos_counts[group] = int(group_df[label_column].sum())
        neg_counts[group] = len(group_df) - pos_counts[group]

    # Target: use the minimum count found across all groups for both classes
    target_pos = min(pos_counts.values())
    target_neg = min(neg_counts.values())

    rows_to_keep = []
    for group in groups:
        group_indices_pos = df[(df[column1] == group) & (df[label_column])].index.tolist()
        group_indices_neg = df[(df[column1] == group) & (~df[label_column])].index.tolist()

        # Randomly sample to match target counts
        rng = np.random.default_rng(42)
        kept_pos = rng.choice(group_indices_pos, size=target_pos, replace=False)
        kept_neg = rng.choice(group_indices_neg, size=target_neg, replace=False)
        rows_to_keep.extend(kept_pos)
        rows_to_keep.extend(kept_neg)

    new_df = df.loc[rows_to_keep].copy()
    removed_count = len(df) - len(new_df)
    return new_df, removed_count


def cap_samples(
    df: pd.DataFrame,
    cap: int,
    label_column: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly samples up to `cap` rows from the DataFrame while maintaining the
    distribution of the `label_column`.
    """
    if len(df) <= cap:
        return df

    # Calculate proportions of each label
    fractions = df[label_column].value_counts(normalize=True)

    rows_to_keep = []
    for label, fraction in fractions.items():
        label_indices = df[df[label_column] == label].index.tolist()
        # Number of samples to keep for this label to maintain distribution
        n_to_keep = round(fraction * cap)
        # Ensure we don't try to keep more than we have (rounding safety)
        n_to_keep = min(n_to_keep, len(label_indices))

        if n_to_keep > 0:
            kept = np.random.RandomState(seed).choice(label_indices, size=n_to_keep, replace=False)
            rows_to_keep.extend(kept)

    # If rounding caused us to have fewer than cap, add some random remaining ones
    if len(rows_to_keep) < cap:
        remaining_indices = list(set(df.index) - set(rows_to_keep))
        n_extra = cap - len(rows_to_keep)
        if remaining_indices and n_extra > 0:
            extra = np.random.RandomState(seed).choice(
                remaining_indices, size=min(n_extra, len(remaining_indices)), replace=False
            )
            rows_to_keep.extend(extra)

    return df.loc[rows_to_keep].copy()


def generate_modification_dict(
    client_ids: int | list[int | str],
    attribute: str,
    value: Any,
    drop_rate_range: tuple[float, float] = (0.0, 0.0),
    flip_rate_range: tuple[float, float] = (0.0, 0.0),
    secondary_attribute: str | None = None,
    secondary_value: Any | None = None,
) -> dict[Any, dict[str, Any]]:
    """
    Generates a modification dictionary for simulating data heterogeneity across clients.
    Linearly interpolates the modification rates across the provided clients.

    Parameters
    ----------
    client_ids : int | list[Any]
        If an int, generates modification for `client_ids` clients (IDs 0 to N-1).
        If a list, generates modification for each client ID in the list.
    attribute : str
        The attribute/column to modify.
    value : Any
        The value of the attribute to target.
    drop_rate_range : tuple[float, float], default=(0.0, 0.0)
        The (start, end) range for the drop rate interpolation.
    flip_rate_range : tuple[float, float], default=(0.0, 0.0)
        The (start, end) range for the flip rate interpolation.
    secondary_attribute : Optional[str], default=None
        Optional second column to condition the modification on.
    secondary_value : Optional[Any], default=None
        Value for the second column condition.

    Returns
    -------
    dict[Any, dict[str, Any]]
        A dictionary mapping client IDs to modification configs.

    """
    clients = list(range(client_ids)) if isinstance(client_ids, int) else client_ids

    num_clients = len(clients)
    mod_dict = {}

    if num_clients <= 1:
        drop_rates = [drop_rate_range[0]]
        flip_rates = [flip_rate_range[0]]
    else:
        drop_step = (drop_rate_range[1] - drop_rate_range[0]) / (num_clients - 1)
        drop_rates = [drop_rate_range[0] + i * drop_step for i in range(num_clients)]

        flip_step = (flip_rate_range[1] - flip_rate_range[0]) / (num_clients - 1)
        flip_rates = [flip_rate_range[0] + i * flip_step for i in range(num_clients)]

    for i, client in enumerate(clients):
        mod_dict[client] = {
            attribute: {
                "drop_rate": drop_rates[i],
                "flip_rate": flip_rates[i],
                "value": value,
                "attribute": secondary_attribute,
                "attribute_value": secondary_value,
            }
        }
    return mod_dict


def generate_bias_by_groups(
    num_total_clients: int, group_configs: list[dict[str, Any]], client_names: list[str | int] | None = None
) -> dict[Any, dict[str, Any]]:
    """
    Generates a modification_dict by partitioning clients into groups and sampling
    bias rates from truncated normal distributions.

    Parameters
    ----------
    num_total_clients : int
        The total number of clients in the federation.
    group_configs : list[dict[str, Any]]
        List of group configurations. Each config should contain:
        - group_id: str
        - num_clients: int
        - sensitive_attr: str
        - sensitive_value: Any
        - intersectional_attr: Optional[str]
        - intersectional_value: Optional[Any]
        - drop_mean: float
        - drop_std: float
        - flip_mean: float
        - flip_std: float
    client_names : Optional[list[str | int]], default=None
        Optional list of names for the clients.

    Returns
    -------
    dict[Any, dict[str, Any]]
        A modification dictionary mapping client IDs to their specific sampled modifications.

    """
    # 1. Validation
    sum_clients = sum(g["num_clients"] for g in group_configs)
    if sum_clients != num_total_clients:
        msg = f"Sum of group clients ({sum_clients}) must equal total clients ({num_total_clients})"
        raise ValueError(msg)

    mod_dict = {}
    client_ids = client_names if client_names else list(range(num_total_clients))

    current_client_idx = 0

    for _g_idx, config in enumerate(group_configs):
        num_g = config["num_clients"]

        # Setup Truncated Normal Generators
        def get_tn_samples(mean, std, n):
            if std <= 0:
                return [mean] * n
            # Bounds for [0, 1]
            a, b = (0 - mean) / std, (1 - mean) / std
            return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)

        drop_rates = get_tn_samples(config["drop_mean"], config["drop_std"], num_g)
        flip_rates = get_tn_samples(config["flip_mean"], config["flip_std"], num_g)

        # Assign to clients
        for i in range(num_g):
            c_id = client_ids[current_client_idx]
            mod_dict[c_id] = {
                config["sensitive_attr"]: {
                    "drop_rate": float(drop_rates[i]),
                    "flip_rate": float(flip_rates[i]),
                    "value": config["sensitive_value"],
                    "attribute": config.get("intersectional_attr"),
                    "attribute_value": config.get("intersectional_value"),
                    "group_id": config["group_id"],
                    "mitigate": config.get("mitigate", False),
                }
            }
            current_client_idx += 1

    return mod_dict


def generate_multiobjective_bias(
    num_total_clients: int,
    group_configs: list[dict[str, Any]],
    client_names: list[str | int] | None = None,
) -> dict[Any, dict[str, Any]]:
    """
    Generates a modification dictionary for multi-objective fairness scenarios.
    Allows defining multiple attribute modifications per group (mitigation or bias injection).

    Parameters
    ----------
    num_total_clients : int
        Total number of clients.
    group_configs : list[dict]
        List of group configurations. Each dict must contain:
        - "group_id": str
        - "num_clients": int
        - "configs": list[dict]
            Each config in the list targets an attribute:
            - "attribute": str (Sensitive attribute name)
            - "mitigate": bool (Optional, defaults to False)
            - "value": Any (Target value for bias injection)
            - "drop_mean", "drop_std": float (For bias injection)
            - "flip_mean", "flip_std": float (For bias injection)
            - "secondary_attribute", "secondary_value": (Optional)
    client_names : list, optional
        List of client names/IDs.

    Returns
    -------
    dict
        Modification dictionary compatible with FairFederatedDataset.

    """
    sum_clients = sum(g["num_clients"] for g in group_configs)
    if sum_clients != num_total_clients:
        msg = f"Sum of group clients ({sum_clients}) must equal total clients ({num_total_clients})"
        raise ValueError(msg)

    mod_dict = {}
    client_ids = client_names if client_names else list(range(num_total_clients))
    current_client_idx = 0

    def get_tn_samples(mean, std, n):
        if std <= 0:
            return [mean] * n
        # Bounds for [0, 1]
        a, b = (0 - mean) / std, (1 - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)

    for group in group_configs:
        num_g = group["num_clients"]
        group_id = group.get("group_id", "Unknown")
        configs = group.get("configs", [])

        # Pre-calculate samples for each config for this group
        # This structure: samples_by_config[config_idx] = {"drop": [...], "flip": [...]}
        samples_by_config = []
        for conf in configs:
            if conf.get("mitigate", False):
                samples_by_config.append(None)  # No sampling needed
            else:
                d_mean = conf.get("drop_mean", 0.0)
                d_std = conf.get("drop_std", 0.0)
                f_mean = conf.get("flip_mean", 0.0)
                f_std = conf.get("flip_std", 0.0)

                samples_by_config.append(
                    {"drop": get_tn_samples(d_mean, d_std, num_g), "flip": get_tn_samples(f_mean, f_std, num_g)}
                )

        # Assign to clients
        for i in range(num_g):
            c_id = client_ids[current_client_idx]
            mod_dict[c_id] = {}

            for idx, conf in enumerate(configs):
                attr = conf["attribute"]

                if conf.get("mitigate", False):
                    mod_dict[c_id][attr] = {"mitigate": True}
                else:
                    samples = samples_by_config[idx]
                    mod_dict[c_id][attr] = {
                        "drop_rate": float(samples["drop"][i]),
                        "flip_rate": float(samples["flip"][i]),
                        "value": conf.get("value"),
                        "attribute": conf.get("secondary_attribute"),
                        "attribute_value": conf.get("secondary_value"),
                        "group_id": group_id,
                    }

            current_client_idx += 1

    return mod_dict
