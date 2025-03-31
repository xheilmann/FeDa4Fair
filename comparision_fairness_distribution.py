
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, Optional, Union, Literal

import matplotlib.colors as mcolors
import pandas as pd
from flwr_datasets.common import EventType, event
from flwr_datasets.metrics.utils import compute_counts, compute_frequencies
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization.bar_plot import _plot_bar
from flwr_datasets.visualization.heatmap_plot import _plot_heatmap
from flwr_datasets.visualization.utils import _validate_parameters

"""Label distribution plotting."""


from typing import Any, Optional, Union

import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flwr_datasets.common import EventType, event
from flwr_datasets.metrics.utils import compute_counts, compute_frequencies
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization.bar_plot import _plot_bar
from flwr_datasets.visualization.heatmap_plot import _plot_heatmap
from flwr_datasets.visualization.utils import _validate_parameters

# pylint: disable=too-many-arguments,too-many-locals

#todo: add all fairness metrics plus model/natural, if model add accuracy if absolut
def plot_fairness_distributions(
    partitioner: Partitioner,
    label_name: str,
    plot_type: str = "heatmap",
    size_unit: Literal["value", "absolute", "attribute-value"] = "absolut",
    max_num_partitions: Optional[int] = None,
    partition_id_axis: str = "x",
    axis: Optional[Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: str = "Per Partition Fairness Distribution",
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    verbose_labels: bool = True,
    plot_kwargs: Optional[dict[str, Any]] = None,
    legend_kwargs: Optional[dict[str, Any]] = None,
    fairness_metric: Literal["DP", "EO"] = "DP",
    model: Optional[Model] = None,

) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot the label distribution of the partitions.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    label_name : str
        Column name identifying label based on which the plot will be created.
    plot_type : str
        Type of plot, either "bar" or "heatmap".
    size_unit : str
        "absolute" or "percent". "absolute" - (number of samples). "percent" -
        normalizes each value, so they sum up to 100%.
    max_num_partitions : Optional[int]
        The number of partitions that will be used. If left None, then all partitions
        will be used.
    partition_id_axis : str
        "x" or "y". The axis on which the partition_id will be marked.
    axis : Optional[Axes]
        Matplotlib Axes object to plot on.
    figsize : Optional[Tuple[float, float]]
        Size of the figure.
    title : str
        Title of the plot.
    cmap : Optional[Union[str, mcolors.Colormap]]
        Colormap for determining the colorspace of the plot.
    legend : bool
        Include the legend.
    legend_title : Optional[str]
        Title for the legend. If None, the defaults will be takes based on the type of
        plot.
    verbose_labels : bool
        Whether to use verbose versions of the labels. These values are used as columns
        of the returned dataframe and as labels on the legend in a bar plot and columns/
        rows ticks in a heatmap plot.
    plot_kwargs: Optional[Dict[str, Any]]
        Any key value pair that can be passed to a plot function that are not supported
        directly. In case of the parameter doubling (e.g. specifying cmap here too) the
        chosen value will be taken from the explicit arguments (e.g. cmap specified as
        an argument to this function not the value in this dictionary).
    legend_kwargs: Optional[Dict[str, Any]]
        Any key value pair that can be passed to a figure.legend in case of bar plot or
        cbar_kws in case of heatmap that are not supported directly. In case of the
        parameter doubling (e.g. specifying legend_title here too) the
        chosen value will be taken from the explicit arguments (e.g. legend_title
        specified as an argument to this function not the value in this dictionary).

    Returns
    -------
    fig : Figure
        The figure object.
    axis : Axes
        The Axes object with the plot.
    dataframe : pd.DataFrame
        The DataFrame where each row represents the partition id and each column
        represents the class.

    Examples
    --------
    Visualize the label distribution resulting from DirichletPartitioner.

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> figure, axis, dataframe = plot_label_distributions(
    >>>     partitioner=partitioner,
    >>>     label_name="label",
    >>>     legend=True,
    >>>     verbose_labels=True,
    >>> )

    Alternatively you can visualize each partition in terms of fraction of the data
    available on that partition instead of the absolute count

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> figure, axis, dataframe = plot_label_distributions(
    >>>     partitioner=partitioner,
    >>>     label_name="label",
    >>>     size_unit="percent",
    >>>     legend=True,
    >>>     verbose_labels=True,
    >>> )
    >>>

    You can also visualize the data as a heatmap by changing the `plot_type` from
    default "bar" to "heatmap"

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> figure, axis, dataframe = plot_label_distributions(
    >>>     partitioner=partitioner,
    >>>     label_name="label",
    >>>     size_unit="percent",
    >>>     plot_type="heatmap",
    >>>     legend=True,
    >>>     plot_kwargs={"annot": True},
    >>> )

    You can also visualize the returned DataFrame in Jupyter Notebook
    >>> dataframe.style.background_gradient(axis=None)
    """
    _validate_parameters(plot_type, size_unit, partition_id_axis)

    #TODO add the fairness metrics here
    if Model not None:
        ...
    if size_unit == "absolute":
        dataframe = compute_counts(
            partitioner=partitioner,
            column_name=label_name,
            verbose_names=verbose_labels,
            max_num_partitions=max_num_partitions,
        )
    elif size_unit == "value":
        dataframe = compute_frequencies(
            partitioner=partitioner,
            column_name=label_name,
            verbose_names=verbose_labels,
            max_num_partitions=max_num_partitions,
        )
    elif size_unit == "attribute-value":
        dataframe = compute_frequencies(
            partitioner=partitioner,
            column_name=label_name,
            verbose_names=verbose_labels,
            max_num_partitions=max_num_partitions,
        )

    axis = _plot_heatmap(
            dataframe,
            axis,
            figsize,
            title,
            cmap,
            partition_id_axis,
            size_unit,
            legend,
            legend_title,
            plot_kwargs,
            legend_kwargs,
        )
    assert axis is not None, "axis is None after plotting"
    figure = axis.figure
    assert isinstance(figure, Figure), "figure extraction from axes is not a Figure"
    return figure, axis, dataframe


def plot_comparison_fairness_distribution(
    partitioner_list: list[Partitioner],
    max_num_partitions: Optional[int] = 30,
    label_name: Union[str, list[str]]= ["SEX", "MAR", "RAC1P"],
    fairness_metric: Literal["DP", "EO"] = "DP",
    size_unit: Literal["value", "absolute", "attribute-value"] = "absolut",
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: Optional[tuple[float, float]] = None,
    subtitle: str = "Fairness Distribution Per Partition",
    titles: Optional[list[str]] = None,
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    verbose_labels: bool = True,
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

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_comparison_label_distribution
    >>>
    >>> partitioner_list = []
    >>> alpha_list = [10_000.0, 100.0, 1.0, 0.1, 0.01, 0.00001]
    >>> for alpha in alpha_list:
    >>>     fds = FederatedDataset(
    >>>         dataset="cifar10",
    >>>         partitioners={
    >>>             "train": DirichletPartitioner(
    >>>                 num_partitions=20,
    >>>                 partition_by="label",
    >>>                 alpha=alpha,
    >>>                 min_partition_size=0,
    >>>             ),
    >>>         },
    >>>     )
    >>>     partitioner_list.append(fds.partitioners["train"])
    >>> fig, axes, dataframe_list = plot_comparison_label_distribution(
    >>>     partitioner_list=partitioner_list,
    >>>     label_name="label",
    >>>     titles=[f"Concentration = {alpha}" for alpha in alpha_list],
    >>> )
    """
    plot_type = "heatmap"
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
            *_, dataframe = plot_fairness_distributions(
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
                fairness_metric=fairness_metric
            )
            dataframe_list.append(dataframe)
        else:
            *_, dataframe = plot_fairness_distributions(
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
                fairness_metric=fairness_metric
            )
            dataframe_list.append(dataframe)

    # Do not use the xlabel and ylabel on each subplot plot
    # (instead use global = per figure xlabel and ylabel)
    for idx, axis in enumerate(axes):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes, partition_id_axis, size_unit)

    # Set up figure xlabel and ylabel
    xlabel, ylabel = _initialize_comparison_xy_labels(
        plot_type, size_unit, partition_id_axis
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    fig.tight_layout()
    return fig, axes, dataframe_list


def _initialize_comparison_figsize(
    figsize: Optional[tuple[float, float]], num_partitioners: int
) -> tuple[float, float]:
    if figsize is not None:
        return figsize
    x_value = 4 + (num_partitioners - 1) * 2
    y_value = 4.8
    figsize = (x_value, y_value)
    return figsize


def _initialize_comparison_xy_labels(
    plot_type: Literal["bar", "heatmap"],
    size_unit: Literal["percent", "absolute"],
    partition_id_axis: Literal["x", "y"],
) -> tuple[str, str]:
    if plot_type == "bar":
        xlabel = "Partition ID"
        ylabel = "Class distribution" if size_unit == "percent" else "Class Count"
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = "Label"
    else:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        )

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel


def _initialize_axis_sharing(
    size_unit: Literal["percent", "absolute"],
    plot_type: Literal["bar", "heatmap"],
    partition_id_axis: Literal["x", "y"],
) -> dict[str, bool]:
    # Do not intervene when the size_unit is percent and plot_type is heatmap
    if size_unit == "percent":
        return {}
    if plot_type == "heatmap":
        return {}
    if partition_id_axis == "x":
        return {"sharey": True}
    if partition_id_axis == "y":
        return {"sharex": True}
    return {"sharex": False, "sharey": False}


def _set_tick_on_value_axes(
    axes: list[Axes],
    partition_id_axis: Literal["x", "y"],
    size_unit: Literal["percent", "absolute"],
) -> None:
    if partition_id_axis == "x" and size_unit == "absolute":
        # Exclude this case due to sharing of y-axis (and thus y-ticks)
        # They must remain set and the number are displayed only on the first plot
        pass
    else:
        for axis in axes[1:]:
            axis.set_yticks([])