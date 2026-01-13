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

import contextlib
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from FeDa4Fair.metrics.fairness import compute_fairness

MAX_LABELS_FOR_BAR_LABEL = 50


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
    legend_kwargs: dict[Any, Any] | None = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """
    Compare the label_name distribution across multiple partitioners.

    Parameters
    ----------
    partitioner_list : list[Partitioner]
        List of partitioners to compare.
    label_name : str | list[str]
        Name of the label column or list of names matching the partitioners.
    plot_type : Literal["bar", "heatmap"], default="bar"
        The type of plot to generate.
    size_unit : Literal["percent", "absolute"], default="percent"
        Whether to show percentages or absolute counts.
    max_num_partitions : int, optional
        Maximum number of partitions to display.
    partition_id_axis : Literal["x", "y"], default="y"
        The axis on which to display the partition IDs.
    figsize : tuple[float, float], optional
        Figure size.
    subtitle : str, default="Comparison of Per Partition Label Distribution"
        Subtitle for the entire figure.
    titles : list[str], optional
        List of titles for each subplot.
    cmap : str | mcolors.Colormap, optional
        Colormap for the plot.
    legend : bool, default=False
        Whether to display a legend.
    legend_title : str, optional
        Title for the legend.
    verbose_labels : bool, default=False
        Whether to use verbose labels from ClassLabel.
    plot_kwargs_list : list[dict[str, Any]], optional
        List of keyword arguments for each plot.
    legend_kwargs : dict[Any, Any], optional
        Keyword arguments for the legend.

    Returns
    -------
    tuple[Figure, list[Axes], list[pd.DataFrame]]
        The figure, axes, and DataFrames used for plotting.

    """
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
    fig, axes = plt.subplots(  # type: ignore[assignment]
        nrows=1,
        ncols=num_partitioners,
        figsize=figsize,
        layout="constrained",
        **axes_sharing,
    )

    # Ensure axes is iterable even if there is only one subplot
    axes_list = [axes] if num_partitioners == 1 else list(axes)

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]

    effective_plot_kwargs_list = [None] * num_partitioners if plot_kwargs_list is None else plot_kwargs_list

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
    sens_att: str | list[str],
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    max_num_partitions: int | None = None,
    partition_id_axis: str = "x",
    axis: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Per Partition Fairness Distribution",
    cmap: str | mcolors.Colormap | None = None,
    legend: bool = False,
    plot_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[Any, Any] | None = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    model: Any | None = None,
    sens_cols: list[str] | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """
    Plot fairness metric distributions across dataset partitions.

    Parameters
    ----------
    partitioner : Partitioner
        The partitioner for training/reference data.
    partitioner_test : Partitioner
        The partitioner for testing data.
    label_name : str
        Name of the label column.
    sens_att : str | list[str]
        Sensitive attribute(s) to evaluate.
    size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
        The level of detail for the metric.
    max_num_partitions : int, optional
        Maximum number of partitions to evaluate.
    partition_id_axis : str, default="x"
        Axis for partition IDs.
    axis : Axes, optional
        Matplotlib axis to plot on.
    figsize : tuple[float, float], optional
        Figure size.
    title : str, default="Per Partition Fairness Distribution"
        Title for the plot.
    cmap : str | mcolors.Colormap, optional
        Colormap for the heatmap.
    legend : bool, default=False
        Whether to show the legend.
    plot_kwargs : dict[str, Any], optional
        Arguments for the heatmap plot.
    legend_kwargs : dict[Any, Any], optional
        Arguments for the legend.
    fairness_metric : Literal["DP", "EO"], default="DP"
        The fairness metric to compute.
    model : Any, optional
        Model to evaluate. If None, evaluates data bias.
    sens_cols : list[str], optional
        Sensitive columns to drop during model training.

    Returns
    -------
    tuple[Figure, Axes, pd.DataFrame]
        The figure, axis, and DataFrame with results.

    """
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
    legend_kwargs: dict[Any, Any] | None = None,
    model: Any | None = None,
    intersectional_fairness: list[str] | None = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """
    Compare fairness metric distributions across multiple partitioners.

    Parameters
    ----------
    partitioner_dict : dict[str, Partitioner]
        Dictionary mapping split names to partitioners.
    max_num_partitions : int, default=30
        Maximum number of partitions to display.
    label_name : str, default="ECP"
        Name of the label column.
    sens_att : str, default="SEX"
        Sensitive attribute to evaluate.
    sens_cols : str | list[str], optional
        Columns to drop from features during model training.
    fairness_metric : Literal["DP", "EO"], default="DP"
        The fairness metric to compute.
    size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
        The level of detail for the metric.
    partition_id_axis : Literal["x", "y"], default="y"
        Axis for partition IDs.
    figsize : tuple[float, float], optional
        Figure size.
    subtitle : str, default="Fairness Distribution Per Partition"
        Subtitle for the entire figure.
    titles : list[str], optional
        Titles for individual subplots.
    cmap : str | mcolors.Colormap, optional
        Colormap for the heatmaps.
    legend : bool, default=False
        Whether to show the legend.
    plot_kwargs_list : list[dict[str, Any]], optional
        Arguments for each heatmap.
    legend_kwargs : dict[Any, Any], optional
        Arguments for the legend.
    model : Any, optional
        Model to evaluate. If None, evaluates data bias.
    intersectional_fairness : list[str], optional
        Attributes for intersectional evaluation.

    Returns
    -------
    tuple[Figure, list[Axes], list[pd.DataFrame]]
        The figure, list of axes, and DataFrames with results.

    """
    eff_sens_cols = [sens_cols] if isinstance(sens_cols, str) else (sens_cols or ["SEX", "MAR", "RAC1P"])
    p_list, p_list_val = _prepare_fairness_partitioners(partitioner_dict, model)
    num_p = len(p_list)
    eff_sens_atts = [sens_att] * num_p if isinstance(sens_att, str) else sens_att

    figsize = _initialize_comparison_figsize(figsize, num_p)
    axes_sharing = _initialize_axis_sharing("absolute", "heatmap", partition_id_axis)
    fig, axes = plt.subplots(nrows=1, ncols=num_p, figsize=figsize, layout="constrained", **axes_sharing)  # type: ignore[assignment]
    axes_list = [axes] if num_p == 1 else list(axes)
    titles = titles or ["" for _ in range(num_p)]
    p_kwargs_list = plot_kwargs_list or [None] * num_p

    df_list = _plot_all_fairness_distributions(
        p_list,
        p_list_val,
        eff_sens_atts,
        intersectional_fairness,
        size_unit,
        partition_id_axis,
        axes_list,
        max_num_partitions,
        cmap,
        legend,
        p_kwargs_list,
        legend_kwargs,
        fairness_metric,
        model,
        eff_sens_cols,
        label_name,
    )

    for idx, axis in enumerate(axes_list):
        axis.set_xlabel(""), axis.set_ylabel(""), axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes_list, partition_id_axis, "absolute")

    xlabel, ylabel = _initialize_comparison_xy_labels("heatmap", "absolute", partition_id_axis, eff_sens_atts)
    fig.supxlabel(xlabel), fig.supylabel(ylabel), fig.suptitle(subtitle)
    return fig, axes_list, df_list


def _prepare_fairness_partitioners(partitioner_dict, model):
    if model is None:
        p_list = list(partitioner_dict.values())
        return p_list, p_list
    p_list = [v for k, v in partitioner_dict.items() if "train" in k]
    p_list_val = [v for k, v in partitioner_dict.items() if "val" in k]
    return p_list, p_list_val


def plot_multi_attribute_fairness(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    label_name: str,
    sens_atts: list[str],
    fairness_metric: Literal["DP", "EO"] = "DP",
    max_num_partitions: int | None = None,
    model: Any | None = None,
    size_unit: Literal["value", "attribute"] = "attribute",
    fds: Any | None = None,
    split: str | None = None,
    test_split: str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str | list[str] | None = None,
    legend: bool = True,
    value_colors: dict[Any, str] | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """
    Plot fairness metrics for multiple sensitive attributes side-by-side for each partition.

    When `size_unit='value'`, it shows two bars per attribute representing the bias toward each group.

    Parameters
    ----------
    partitioner : Partitioner
        The partitioner for training/reference data.
    partitioner_test : Partitioner
        The partitioner for testing data.
    label_name : str
        Name of the label column.
    sens_atts : list[str]
        List of sensitive attributes to evaluate independently.
    fairness_metric : Literal["DP", "EO"], default="DP"
        The fairness metric to compute.
    max_num_partitions : int, optional
        Maximum number of partitions to evaluate.
    model : Any, optional
        The model to evaluate. If None, evaluates data bias.
    size_unit : Literal["value", "attribute"], default="attribute"
        The level of detail for the metric. 'value' enables dual bars and colored bias direction.
    fds : Any, optional
        FairFederatedDataset instance to use for partition loading.
    split : str, optional
        Split name for the training data.
    test_split : str, optional
        Split name for the testing data.
    figsize : tuple[float, float], optional
        Figure size.
    title : str, optional
        Plot title.
    cmap : str | list[str], optional
        Colors for the bars.
    legend : bool, default=True
        Whether to show the legend.
    value_colors : dict[Any, str], optional
        Mapping of attribute values to colors (e.g., {0: 'red', 1: 'blue'}) for 'value' level.
    **plot_kwargs
        Additional arguments passed to the plotting function.

    Returns
    -------
    tuple[Figure, Axes, pd.DataFrame]
        The figure, axis, and combined DataFrame with results.

    """
    from FeDa4Fair.metrics.fairness import compute_multi_fairness

    combined_df = compute_multi_fairness(
        partitioner=partitioner,
        partitioner_test=partitioner_test,
        model=model,
        sens_atts=sens_atts,
        fairness_metric=fairness_metric,
        label_name=label_name,
        max_num_partitions=max_num_partitions,
        size_unit="attribute-value" if size_unit == "value" else size_unit,
        fds=fds,
        split=split,
        test_split=test_split,
    )

    if size_unit == "value":
        plot_df, bar_colors = _prepare_value_plot_data(combined_df, sens_atts, value_colors)
    else:
        metric_cols = [f"{attr}_{fairness_metric}" for attr in sens_atts]
        plot_df = combined_df[metric_cols].copy()
        plot_df.columns = sens_atts
        bar_colors = [cmap] * len(sens_atts) if isinstance(cmap, str) else (cmap or [None] * len(sens_atts))

    figsize = figsize or (max(8.0, len(plot_df) * 0.5), 6.0)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _configure_multi_attribute_axis(ax, plot_df, title, fairness_metric, bar_colors, **plot_kwargs)

    if legend:
        _add_multi_attribute_legend(ax, size_unit, value_colors)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return fig, ax, combined_df


def _configure_multi_attribute_axis(ax, plot_df, title, metric, bar_colors, **plot_kwargs):
    """
    Configure axis labels and titles for multi-attribute plot.

    Parameters
    ----------
    ax : Axes
        The matplotlib axis to configure.
    plot_df : pd.DataFrame
        DataFrame containing metrics to plot.
    title : str, optional
        The title for the plot.
    metric : str
        The name of the metric (DP or EO).
    bar_colors : list[str] | list[list[str]]
        Colors for the individual bars.
    **plot_kwargs
        Keyword arguments for the bar plot.

    """
    ax.set_title(title or f"{metric} by Attribute per Partition")
    x = np.arange(len(plot_df))
    width = 0.8 / len(plot_df.columns)
    for i, attr in enumerate(plot_df.columns):
        pos = x - 0.4 + (i + 0.5) * width
        color = bar_colors[i] if isinstance(bar_colors, list) and i < len(bar_colors) else None
        bars = ax.bar(pos, plot_df[attr], width, label=attr, color=color, **plot_kwargs)
        if len(plot_df) * len(plot_df.columns) < MAX_LABELS_FOR_BAR_LABEL:
            ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel(f"{metric} Difference")
    ax.set_xlabel("Partition ID")


def _add_multi_attribute_legend(ax, size_unit, value_colors):
    """
    Add appropriate legend to multi-attribute plot.

    Parameters
    ----------
    ax : Axes
        The matplotlib axis.
    size_unit : str
        The level of detail.
    value_colors : dict[Any, str], optional
        Custom color mapping for values.

    """
    if size_unit == "value" and value_colors:
        val_handles = [mpatches.Patch(color=color, label=f"Bias toward {val}") for val, color in value_colors.items()]
        ax.legend(handles=val_handles, title="Legend")
    else:
        ax.legend(title="Sensitive Attribute")


def _prepare_value_plot_data(combined_df, sens_atts, value_colors):
    plot_data = {}
    all_bar_colors = []
    for attr in sens_atts:
        pattern = f"{attr}_"
        attr_cols = [c for c in combined_df.columns if c.startswith(pattern)]
        plot_data[attr] = combined_df[attr_cols].max(axis=1)
        if value_colors:

            def get_row_color(row):
                if row.max() <= 0:
                    return "gray"
                parts = row.idxmax().split("_")
                group_a = parts[-2]
                with contextlib.suppress(ValueError, TypeError):
                    group_a = int(float(group_a))
                return value_colors.get(group_a, "gray")

            all_bar_colors.append([get_row_color(r) for _, r in combined_df[attr_cols].iterrows()])
    return pd.DataFrame(plot_data), all_bar_colors


def _plot_all_fairness_distributions(
    p_list,
    p_list_val,
    eff_sens_atts,
    intersectional,
    size_unit,
    p_id_axis,
    axes_list,
    max_parts,
    cmap,
    legend,
    p_kwargs_list,
    l_kwargs,
    f_metric,
    model,
    eff_cols,
    label_name,
):
    df_list = []
    for idx, (p, s_att, p_kw, ax) in enumerate(zip(p_list, eff_sens_atts, p_kwargs_list, axes_list, strict=False)):
        target_s_att = intersectional or s_att
        is_last = idx == (len(p_list) - 1)

        _, _, df = plot_fairness_distributions(
            partitioner=p,
            partitioner_test=p_list_val[idx],
            label_name=label_name,
            sens_att=target_s_att,
            size_unit=size_unit,
            max_num_partitions=max_parts,
            partition_id_axis=p_id_axis,
            axis=ax,
            cmap=cmap,
            legend=legend if is_last else False,
            plot_kwargs=p_kw,
            legend_kwargs=l_kwargs,
            fairness_metric=f_metric,
            model=model,
            sens_cols=eff_cols,
        )
        df_list.append(df)
    return df_list


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
