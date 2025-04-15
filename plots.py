# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
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
"""Comparison of label distribution plotting."""


from typing import Any, Literal, Optional, Union

from flwr_datasets.visualization.comparison_label_distribution import _set_tick_on_value_axes, _initialize_comparison_figsize, _initialize_axis_sharing
from flwr_datasets.visualization.heatmap_plot import _plot_heatmap

from fairness_computation import compute_fairness
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization.constants import PLOT_TYPES
from flwr_datasets.visualization.label_distribution import plot_label_distributions


# pylint: disable=too-many-arguments,too-many-locals
# mypy: disable-error-code="call-overload"
def plot_comparison_label_distribution(
    partitioner_list: list[Partitioner],
    label_name: Union[str, list[str]],
    plot_type: Literal["bar", "heatmap"] = "bar",
    size_unit: Literal["percent", "absolute"] = "percent",
    max_num_partitions: Optional[int] = 30,
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: Optional[tuple[float, float]] = None,
    subtitle: str = "Comparison of Per Partition Label Distribution",
    titles: Optional[list[str]] = None,
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    verbose_labels: bool = False,
    plot_kwargs_list: Optional[list[Optional[dict[str, Any]]]] = None,
    legend_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """Compare the label distribution across multiple partitioners.

    Parameters
    ----------
    partitioner_list : List[Partitioner]
        List of partitioners to be compared.
    label_name : Union[str, List[str]]
        Column name or list of column names identifying labels for each partitioner.
    plot_type :  Literal["bar", "heatmap"]
        Type of plot, either "bar" or "heatmap".
    size_unit : Literal["percent", "absolute"]
        "absolute" for raw counts, or "percent" to normalize values to 100%.
    max_num_partitions : Optional[int]
        Maximum number of partitions to include in the plot. If None, all partitions
        are included.
    partition_id_axis : Literal["x", "y"]
        Axis on which the partition IDs will be marked, either "x" or "y".
    figsize : Optional[Tuple[float, float]]
        Size of the figure. If None, a default size is calculated.
    subtitle : str
        Subtitle for the figure. Defaults to "Comparison of Per Partition Label
        Distribution"
    titles : Optional[List[str]]
        Titles for each subplot. If None, no titles are set.
    cmap : Optional[Union[str, mcolors.Colormap]]
        Colormap for determining the colorspace of the plot.
    legend : bool
        Whether to include a legend. If True, it will be included right-hand side after
        all the plots.
    legend_title : Optional[str]
        Title for the legend. If None, the defaults will be takes based on the type of
        plot.
    verbose_labels : bool
        Whether to use verbose versions of the labels.
    plot_kwargs_list: Optional[List[Optional[Dict[str, Any]]]]
        List of plot_kwargs. Any key value pair that can be passed to a plot function
        that are not supported directly. In case of the parameter doubling
        (e.g. specifying cmap here too) the chosen value will be taken from the
        explicit arguments (e.g. cmap specified as an argument to this function not
        the value in this dictionary).
    legend_kwargs: Optional[Dict[str, Any]]
        Any key value pair that can be passed to a figure.legend in case of bar plot or
        cbar_kws in case of heatmap that are not supported directly. In case of
        parameter doubling (e.g. specifying legend_title here too) the
        chosen value will be taken from the explicit arguments (e.g. legend_title
        specified as an argument to this function not the value in this dictionary).

    Returns
    -------
    fig : Figure
        The figure object containing the plots.
    axes_list : List[Axes]
        List of Axes objects for the plots.
    dataframe_list : List[pd.DataFrame]
        List of DataFrames used for each plot.

    Examples
    --------
    Compare the difference of using different alpha (concentration) parameters in
    DirichletPartitioner.

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
        label_name = [label_name] * num_partitioners
    elif isinstance(label_name, list):
        pass
    else:
        raise TypeError(
            f"Label name has to be of type List[str] or str but given "
            f"{type(label_name)}"
        )
    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    axes_sharing = _initialize_axis_sharing(size_unit, plot_type, partition_id_axis)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_partitioners,
        figsize=figsize,
        layout="constrained",
        **axes_sharing,
    )

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]

    if plot_kwargs_list is None:
        plot_kwargs_list = [None] * num_partitioners

    dataframe_list = []
    for idx, (partitioner, single_label_name, plot_kwargs) in enumerate(
        zip(partitioner_list, label_name, plot_kwargs_list)
    ):
        if idx == (num_partitioners - 1):
            *_, dataframe = plot_label_distributions(
                partitioner=partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
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
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
                plot_kwargs=plot_kwargs,
            )
            dataframe_list.append(dataframe)

    # Do not use the xlabel and ylabel on each subplot plot
    # (instead use global = per figure xlabel and ylabel)
    for idx, axis in enumerate(axes):
        axis.set_xlabel(f"")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes, partition_id_axis, size_unit)

    # Set up figure xlabel and ylabel
    xlabel, ylabel = _initialize_comparison_xy_labels(
        plot_type, size_unit, partition_id_axis, label_name
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)


    return fig, axes, dataframe_list

#todo: if model then also plots for accuracy?
def plot_fairness_distributions(
    partitioner: Partitioner,
    partitioner_test: Partitioner,
    class_label: str,
    label_name: str,
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    max_num_partitions: Optional[int] = None,
    partition_id_axis: str = "x",
    axis: Optional[Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: str = "Per Partition Fairness Distribution",
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    legend: bool = False,
    plot_kwargs: Optional[dict[str, Any]] = None,
    legend_kwargs: Optional[dict[str, Any]] = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    model: Optional = None,
    sensitive_attributes: list[str]=None) -> tuple[Figure, Axes, pd.DataFrame]:
    """
    Plot fairness metric distributions across dataset partitions in a partitioner object.

    Visualizes how fairness metrics (Demographic Parity or Equalized Odds)
    vary per partition with respect to one or more sensitive attributes. The function
    supports different granularity levels and can be applied to both training and test
    partitions using a given model (required for EO).

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner containing training partitions used for fairness evaluation.

    partitioner_test : Partitioner
        Partitioner for the test set, needed when computing prediction-based metrics
        such as Equalized Odds.

    class_label : str
        The column name of the label.

    label_name : str
        Column name of the attribute for fairness evaluation.

    size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
        The level at which fairness is evaluated:
        - "attribute": only worst fairness metric is returned,
        - "value": worst fairness metric as well as for which values this fairness was calculated is returned,
        - "attribute-value": all possible fairness metric values are returned.

    max_num_partitions : Optional[int], default=None
        Limit on number of partitions to visualize. If `None`, all partitions are shown.

    partition_id_axis : str, default="x"
        Axis to use for partition IDs. Must be "x" or "y".

    axis : Optional[Axes], default=None
        Matplotlib `Axes` object to plot on. If `None`, a new figure and axes will be created.

    figsize : Optional[Tuple[float, float]], default=None
        Dimensions of the figure in inches.

    title : str, default="Per Partition Fairness Distribution"
        Title of the plot.

    cmap : Optional[Union[str, mcolors.Colormap]], default=None
        Colormap used for plotting fairness distributions.

    legend : bool, default=False
        Whether to include a legend in the plot.

    plot_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword arguments passed to the plotting function.

    legend_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword arguments for customizing the legend or colorbar.

    fairness_metric : Literal["DP", "EO"], default="DP"
        Fairness metric to compute:
        - "DP": Demographic Parity
        - "EO": Equalized Odds (requires model)

    model : Optional, default=None
        Model which will be trained and then used for generating predictions, required for EO metric.

    sensitive_attributes : list of str, default=None
        List of sensitive attribute columns to take out if training on a model.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.

    axis : Axes
        The axes object containing the plot.

    dataframe : pd.DataFrame
        DataFrame where rows correspond to partitions and columns correspond
        to what is specified in size_unit and their fairness scores.
    """


    dataframe = compute_fairness( partitioner= partitioner,
                                  partitioner_test=partitioner_test,
                                  model=model,
                                  column_name=label_name,
                                  fairness_metric=fairness_metric,
                                  label=class_label,
                                  max_num_partitions=max_num_partitions,
                                  sensitive_attributes=sensitive_attributes,
                                  size_unit =size_unit)




    if size_unit in ["attribute","value"]:
        plot_kwargs = {"annot": dataframe.drop(f"{label_name}_{fairness_metric}", axis=1)}
        if size_unit == "attribute":
            plot_kwargs["fmt"] = ".2f"
        else:
            plot_kwargs["fmt"] = "s"
        dataframe = dataframe.drop(f"{label_name}_val", axis=1)
    elif len(dataframe.columns)< 6:
        plot_kwargs = {"annot": True, "fmt": ".2f"}
    axis = _plot_heatmap(
            dataframe,
            axis,
            figsize,
            title,
            cmap,
            partition_id_axis,
            "absolute",
            legend,
            fairness_metric,
            plot_kwargs,
            legend_kwargs,
        )
    assert axis is not None, "axis is None after plotting"
    figure = axis.figure
    assert isinstance(figure, Figure), "figure extraction from axes is not a Figure"
    return figure, axis, dataframe


def plot_comparison_fairness_distribution(
    partitioner_dict: dict[str,Partitioner],
    max_num_partitions: Optional[int] = 30,
    class_label: str = "ECP",
    label_name: Union[str, list[str]]= ["SEX", "MAR", "RAC1P"],
    fairness_metric: Literal["DP", "EO"] = "DP",
    size_unit: Literal["value", "attribute", "attribute-value"] = "attribute",
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: Optional[tuple[float, float]] = None,
    subtitle: str = "Fairness Distribution Per Partition",
    titles: Optional[list[str]] = None,
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    verbose_labels: bool = False,
    plot_kwargs_list: Optional[list[Optional[dict[str, Any]]]] = None,
    legend_kwargs: Optional[dict[str, Any]] = None,
    model: Optional = None,
    intersectional_fairness: list[str] = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """
     Compare fairness metric distributions across multiple partitioners.

     This function plots the fairness distribution (Demographic Parity or Equalized Odds)
     per partition for multiple datasets or clients. Each subplot corresponds to a
     different partitioner and visualizes fairness across partitions using heatmaps.

     Parameters
     ----------
     partitioner_dict : dict[str, Partitioner]
         Dictionary mapping names (e.g., states) to partitioners containing the
         respective dataset partitions.

     max_num_partitions : Optional[int], default=30
         Maximum number of partitions to include per partitioner. If None, all partitions
         are included.

     class_label : str, default="ECP"
         The name of the label.

     label_name : Union[str, list[str]], default=["SEX", "MAR", "RAC1P"]
         Sensitive attribute(s) to compute fairness on. If a list is provided, fairness
         is computed on each of the labels except if intersectional fairness parameter is specified.
         Then only intersectional fairness is computed.

     fairness_metric : Literal["DP", "EO"], default="DP"
         The fairness metric to use:
         - "DP": Demographic Parity
         - "EO": Equalized Odds (requires predictions from a model)

     size_unit : Literal["value", "attribute", "attribute-value"], default="attribute"
         The level at which fairness is evaluated:
        - "attribute": only worst fairness metric is returned,
        - "value": worst fairness metric as well as for which values this fairness was calculated is returned,
        - "attribute-value": all possible fairness metric values are returned.

     partition_id_axis : Literal["x", "y"], default="y"
         The axis on which partition IDs are placed.

     figsize : Optional[tuple[float, float]], default=None
         Size of the overall figure. If None, a default size is used.

     subtitle : str, default="Fairness Distribution Per Partition"
         Subtitle displayed at the top of the entire figure.

     titles : Optional[list[str]], default=None
         List of subplot titles corresponding to each partitioner. If None, default names
         from `partitioner_dict` keys are used.

     cmap : Optional[Union[str, mcolors.Colormap]], default=None
         Colormap used for plotting fairness values.

     legend : bool, default=False
         Whether to display a legend next to the plots.

     legend_title : Optional[str], default=None
         Title for the legend.

     verbose_labels : bool, default=False
         Whether to display full textual labels for sensitive attribute values.

     plot_kwargs_list : Optional[list[Optional[dict[str, Any]]]], default=None
         List of dictionaries with extra plotting arguments for each subplot.

     legend_kwargs : Optional[dict[str, Any]], default=None
         Additional keyword arguments for customizing the legend or colorbar.

     model : Optional, default=None
         Model to train which has an implemented .fit() used to compute predictions for fairness evaluation.

     intersectional_fairness : list[str], default=None
         If provided, fairness will be only computed on the intersection of these sensitive
         attributes.

     Returns
     -------
     fig : Figure
         The matplotlib Figure object containing the full comparison plot.

     axes_list : list[Axes]
         List of Axes objects for each subplot (one per partitioner).

     dataframe_list : list[pd.DataFrame]
         List of DataFrames containing fairness metrics per partition for each partitioner.
     """
    global partitioner_list_val
    plot_type = "heatmap"
    if model is None:
        partitioner_list = list(partitioner_dict.values())
        partitioner_list_val = partitioner_list
    else:
        partitioner_list = [value for key, value in partitioner_dict.items() if "train" in key]
        partitioner_list_val =  [value for key, value in partitioner_dict.items() if "val" in key]
    num_partitioners = len(partitioner_list)
    if isinstance(label_name, str):
        label_name = [label_name] * num_partitioners
    elif isinstance(label_name, list):
        pass
    else:
        raise TypeError(
            f"Label name has to be of type List[str] or str but given "
            f"{type(label_name)}"
        )
    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    axes_sharing = _initialize_axis_sharing(size_unit, plot_type, partition_id_axis)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_partitioners,
        figsize=figsize,
        layout="constrained",
        **axes_sharing,
    )

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]

    if plot_kwargs_list is None:
        plot_kwargs_list = [None] * num_partitioners

    dataframe_list = []
    sens_att = label_name

    for idx, (partitioner, single_label_name, plot_kwargs) in enumerate(
        zip(partitioner_list, label_name, plot_kwargs_list)
    ):

        if intersectional_fairness is not None:
            single_label_name = intersectional_fairness
        if idx == (num_partitioners - 1):
            *_, dataframe = plot_fairness_distributions(
                partitioner=partitioner,
                partitioner_test = partitioner_list_val[idx],
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=legend,
                legend_title=legend_title,
                verbose_labels=verbose_labels,
                plot_kwargs=plot_kwargs,
                legend_kwargs=legend_kwargs,
                fairness_metric=fairness_metric,
                model = model,
                sensitive_attributes = sens_att,
                class_label=class_label,

            )
            dataframe_list.append(dataframe)
        else:
            *_, dataframe = plot_fairness_distributions(
                partitioner=partitioner,
                partitioner_test=partitioner_list_val[idx],
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
                plot_kwargs=plot_kwargs,
                fairness_metric=fairness_metric,
                model=model,
                sensitive_attributes=sens_att,
                class_label=class_label,


            )
            dataframe_list.append(dataframe)

    for idx, axis in enumerate(axes):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes, partition_id_axis, size_unit)

    xlabel, ylabel = _initialize_comparison_xy_labels(
        plot_type, size_unit, partition_id_axis, label_name
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    return fig, axes, dataframe_list


def _initialize_comparison_xy_labels(
    plot_type: Literal["bar", "heatmap"],
    size_unit: Literal["percent", "absolute"],
    partition_id_axis: Literal["x", "y"],
    label_name: str,
) -> tuple[str, str]:
    """
    modified so that ylabel fits the label_names
    """
    if plot_type == "bar":
        xlabel = "Partition ID"
        ylabel = "Class distribution" if size_unit == "percent" else "Class Count"
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = label_name[0]
    else:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        )

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel








