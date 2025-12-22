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

"""Functions for plotting fairness metrics and label distribution."""

from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from fairness_computation import compute_fairness
from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization.comparison_label_distribution import (
    _initialize_axis_sharing,
    _initialize_comparison_figsize,
    _set_tick_on_value_axes,
)
from flwr_datasets.visualization.constants import PLOT_TYPES
from flwr_datasets.visualization.heatmap_plot import _plot_heatmap
from flwr_datasets.visualization.label_distribution import plot_label_distributions
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_comparison_label_distribution(
    partitioner_list: list[Partitioner],
    label_name: str | list[str],
    plot_type: Literal["bar", "heatmap"] = "bar",
    size_unit: Literal["percent", "absolute"] = "percent",
    max_num_partitions: int | None = 30,
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: tuple[float, float] | None = None,
    subtitle: str = "Comparison of Per Partition Label Distribution",
    titles: list[str] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    verbose_labels: bool = False,
    plot_kwargs_list: list[dict[str, Any] | None] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """Compare the label_name distribution across multiple partitioners."""
    event(
        EventType.PLOT_COMPARISON_LABEL_DISTRIBUTION_CALLED,
        {
            "num_compare": len(partitioner_list),
            "plot_type": plot_type,
        },
    )
    num_partitioners = len(partitioner_list)
    if isinstance(label_name, str):
        effective_label_names = [label_name] * num_partitioners
    elif isinstance(label_name, list):
        effective_label_names = label_name
    else:
        msg = f"Label name has to be of type List[str] or str but given {type(label_name)}"
        raise TypeError(msg)

    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    axes_sharing = _initialize_axis_sharing(size_unit, plot_type, partition_id_axis)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_partitioners,
        figsize=figsize,
        layout="constrained",
        **axes_sharing,
    )

    # Ensure axes is iterable even if there is only one subplot
    if num_partitioners == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]

    if plot_kwargs_list is None:
        effective_plot_kwargs_list = [None] * num_partitioners
    else:
        effective_plot_kwargs_list = plot_kwargs_list

    dataframe_list = []
    for idx, (partitioner, single_label_name, plot_kwargs) in enumerate(
        zip(partitioner_list, effective_label_names, effective_plot_kwargs_list, strict=False)
    ):
        if idx == (num_partitioners - 1):
            *_, dataframe = plot_label_distributions(
                partitioner=partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes_list[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=legend,
                legend_title=legend_title,
                verbose_labels=verbose_labels,
                plot_kwargs=plot_kwargs,
                legend_kwargs=legend_kwargs,
            )
            dataframe_list.append(dataframe)
        else:
            *_, dataframe = plot_label_distributions(
                partitioner=partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes_list[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
                plot_kwargs=plot_kwargs,
            )
            dataframe_list.append(dataframe)

    for idx, axis in enumerate(axes_list):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes_list, partition_id_axis, size_unit)

    xlabel, ylabel = _initialize_comparison_xy_labels(plot_type, size_unit, partition_id_axis, effective_label_names)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    return fig, axes_list, dataframe_list


def plot_fairness_distributions(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    label_name: str,
    sens_att: str,
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    max_num_partitions: int | None = None,
    partition_id_axis: str = "x",
    axis: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Per Partition Fairness Distribution",
    cmap: str | mcolors.Colormap | None = None,
    legend: bool = False,
    plot_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    model: Any | None = None,
    sens_cols: list[str] | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot fairness metric distributions across dataset partitions."""
    dataframe = compute_fairness(
        partitioner=partitioner,
        partitioner_test=partitioner_test,
        model=model,
        sens_att=sens_att,
        fairness_metric=fairness_metric,
        label_name=label_name,
        max_num_partitions=max_num_partitions,
        sens_cols=sens_cols,
        size_unit=size_unit,
    )

    effective_plot_kwargs = plot_kwargs.copy() if plot_kwargs is not None else {}
    if size_unit in ["attribute", "value"]:
        effective_plot_kwargs["annot"] = dataframe.drop(f"{sens_att}_{fairness_metric}", axis=1)
        if size_unit == "attribute":
            effective_plot_kwargs["fmt"] = ".2f"
        else:
            effective_plot_kwargs["fmt"] = "s"
        dataframe = dataframe.drop(f"{sens_att}_val", axis=1)
    elif len(dataframe.columns) < 6:  # noqa: PLR2004
        effective_plot_kwargs["annot"] = True
        effective_plot_kwargs["fmt"] = ".2f"

    effective_plot_kwargs["vmin"] = 0
    effective_plot_kwargs["vmax"] = 1
    effective_plot_kwargs["cmap"] = "Spectral_r"
    effective_plot_kwargs["annot_kws"] = {"fontsize": 14}

    res_axis = _plot_heatmap(
        dataframe,
        axis,
        figsize,
        title,
        cmap,
        partition_id_axis,
        "absolute",
        legend,
        fairness_metric,
        effective_plot_kwargs,
        legend_kwargs,
    )
    if res_axis is None:
        msg = "axis is None after plotting"
        raise ValueError(msg)
    figure = res_axis.figure
    if not isinstance(figure, Figure):
        msg = "figure extraction from axes is not a Figure"
        raise TypeError(msg)
    return figure, res_axis, dataframe


def plot_comparison_fairness_distribution(
    partitioner_dict: dict[str, Partitioner],
    max_num_partitions: int | None = 30,
    label_name: str = "ECP",
    sens_att: str = "SEX",
    sens_cols: str | list[str] | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: tuple[float, float] | None = None,
    subtitle: str = "Fairness Distribution Per Partition",
    titles: list[str] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    legend: bool = False,
    plot_kwargs_list: list[dict[str, Any] | None] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    model: Any | None = None,
    intersectional_fairness: list[str] | None = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """Compare fairness metric distributions across multiple partitioners."""
    if sens_cols is None:
        effective_sens_cols = ["SEX", "MAR", "RAC1P"]
    else:
        effective_sens_cols = [sens_cols] if isinstance(sens_cols, str) else sens_cols

    plot_type = "heatmap"
    if model is None:
        partitioner_list = list(partitioner_dict.values())
        partitioner_list_val = partitioner_list
    else:
        partitioner_list = [value for key, value in partitioner_dict.items() if "train" in key]
        partitioner_list_val = [value for key, value in partitioner_dict.items() if "val" in key]

    num_partitioners = len(partitioner_list)
    if isinstance(sens_att, str):
        effective_sens_atts = [sens_att] * num_partitioners
    elif isinstance(sens_att, list):
        effective_sens_atts = sens_att
    else:
        msg = f"Label name has to be of type List[str] or str but given {type(sens_att)}"
        raise TypeError(msg)

    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    axes_sharing = _initialize_axis_sharing(size_unit, plot_type, partition_id_axis)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_partitioners,
        figsize=figsize,
        layout="constrained",
        **axes_sharing,
    )

    if num_partitioners == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]

    if plot_kwargs_list is None:
        effective_plot_kwargs_list = [None] * num_partitioners
    else:
        effective_plot_kwargs_list = plot_kwargs_list

    dataframe_list = []

    for idx, (partitioner, single_sens_att, plot_kwargs) in enumerate(
        zip(partitioner_list, effective_sens_atts, effective_plot_kwargs_list, strict=False)
    ):
        if intersectional_fairness is not None:
            # Note: PLW2901 warning might trigger if we reassign loop var, but we use it only locally
            target_sens_att = intersectional_fairness
        else:
            target_sens_att = single_sens_att

        if idx == (num_partitioners - 1):
            *_, dataframe = plot_fairness_distributions(
                partitioner=partitioner,
                partitioner_test=partitioner_list_val[idx],
                sens_att=target_sens_att,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes_list[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=legend,
                plot_kwargs=plot_kwargs,
                legend_kwargs=legend_kwargs,
                fairness_metric=fairness_metric,
                model=model,
                sens_cols=effective_sens_cols,
                label_name=label_name,
            )
            dataframe_list.append(dataframe)
        else:
            *_, dataframe = plot_fairness_distributions(
                partitioner=partitioner,
                partitioner_test=partitioner_list_val[idx],
                sens_att=target_sens_att,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes_list[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
                plot_kwargs=plot_kwargs,
                fairness_metric=fairness_metric,
                model=model,
                sens_cols=effective_sens_cols,
                label_name=label_name,
            )
            dataframe_list.append(dataframe)

    for idx, axis in enumerate(axes_list):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes_list, partition_id_axis, size_unit)

    xlabel, ylabel = _initialize_comparison_xy_labels(plot_type, size_unit, partition_id_axis, effective_sens_atts)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    return fig, axes_list, dataframe_list


def _initialize_comparison_xy_labels(
    plot_type: Literal["bar", "heatmap"],
    size_unit: Literal["percent", "absolute"],
    partition_id_axis: Literal["x", "y"],
    label_name: list[str],
) -> tuple[str, str]:
    """Initialize comparison xy labels."""
    if plot_type == "bar":
        xlabel = "Partition ID"
        ylabel = "Class distribution" if size_unit == "percent" else "Class Count"
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = label_name[0]
    else:
        msg = f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        raise ValueError(msg)

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel
