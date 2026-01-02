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
    Supports value-based color coding for a single bar per attribute when size_unit='value'.
    """
    from FeDa4Fair.metrics.fairness import compute_multi_fairness

    if size_unit == "value":
        # Use 'attribute-value' to get all pairwise differences for coloring logic
        combined_df = compute_multi_fairness(
            partitioner=partitioner,
            partitioner_test=partitioner_test,
            model=model,
            sens_atts=sens_atts,
            fairness_metric=fairness_metric,
            label_name=label_name,
            max_num_partitions=max_num_partitions,
            size_unit="attribute-value",
            fds=fds,
            split=split,
            test_split=test_split
        )

        plot_data = {}
        all_bar_colors = []

        for attr in sens_atts:
            pattern = f"{attr}_"
            attr_cols = [c for c in combined_df.columns if c.startswith(pattern)]
            
            # The magnitude is the max disparity
            plot_data[attr] = combined_df[attr_cols].max(axis=1)

            if value_colors:
                def get_row_color(row):
                    if row.max() <= 0:
                        return "gray"
                    best_col = row.idxmax()
                    parts = best_col.split("_")
                    # Group responsible is the second to last part
                    group_a = parts[-2]
                    try:
                        group_a = int(float(group_a))
                    except (ValueError, TypeError):
                        pass
                    return value_colors.get(group_a, "gray")

                attr_colors = [get_row_color(r) for _, r in combined_df[attr_cols].iterrows()]
                all_bar_colors.append(attr_colors)
        
        plot_df = pd.DataFrame(plot_data)
        bar_colors = all_bar_colors
    else:
        combined_df = compute_multi_fairness(
            partitioner=partitioner,
            partitioner_test=partitioner_test,
            model=model,
            sens_atts=sens_atts,
            fairness_metric=fairness_metric,
            label_name=label_name,
            max_num_partitions=max_num_partitions,
            size_unit=size_unit,
            fds=fds,
            split=split,
            test_split=test_split
        )
        metric_cols = [f"{attr}_{fairness_metric}" for attr in sens_atts]
        plot_df = combined_df[metric_cols].copy()
        plot_df.columns = sens_atts
        bar_colors = [cmap] * len(sens_atts) if isinstance(cmap, str) else (cmap or [None] * len(sens_atts))

    if figsize is None:
        num_partitions = len(plot_df)
        figsize = (max(8.0, num_partitions * 0.5), 6.0)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    
    if title is None:
        title = f"{fairness_metric} by Attribute per Partition"

    x = np.arange(len(plot_df))
    width = 0.8 / len(sens_atts)
    
    for i, attr in enumerate(sens_atts):
        pos = x - 0.4 + (i + 0.5) * width
        color = bar_colors[i] if isinstance(bar_colors, list) and i < len(bar_colors) else None
        bars = ax.bar(pos, plot_df[attr], width, label=attr, color=color, **plot_kwargs)
        if len(plot_df) * len(sens_atts) < 50:
            ax.bar_label(bars, fmt='%.2f', padding=3)

    ax.set_title(title)
    ax.set_ylabel(f"{fairness_metric} Difference")
    ax.set_xlabel("Partition ID")
    
    if legend:
        if size_unit == "value" and value_colors:
            import matplotlib.patches as mpatches
            val_handles = [mpatches.Patch(color=color, label=f"Bias toward {val}") for val, color in value_colors.items()]
            ax.legend(handles=val_handles, title="Legend")
        else:
            ax.legend(title="Sensitive Attribute")
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig, ax, combined_df


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
