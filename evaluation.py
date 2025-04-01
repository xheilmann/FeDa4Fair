#TODO: per client plot per attribute/feature value with before and after Fl and for the before the input is a numpy array from the dataset generation (accuracy, fairness)
# 6. output
# - dataset statistics per client(  # datapoints, fairness metrics (gender, race mar), performance metrics, modifications)
# -overall statistics global model FedAVG((  # datapoints, fairness metrics(gender, race mar), performance metrics) before modifications and after
# - global model on pooled dataset


from flwr_datasets.visualization import plot_comparison_label_distribution
from pandas import DataFrame

from comparision_fairness_distribution import plot_comparison_fairness_distribution
from typing import Any, Optional, Union, Literal

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


def evaluate_fairness(
    partitioner_dict: dict[str, Partitioner],
    max_num_partitions: Optional[int] = 30,
    label_name: Union[str, list[str]]= ["SEX", "MAR", "RAC1P"],
    size_unit: Literal["percent", "absolute"] = "absolut",
    fairness_metric: Literal["DP", "EO"] = "DP",
    fairness: Literal["attribute", "value","attribute-value"] = "attribute",
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
    model: Optional = None,
    class_label: str="ECP",
) -> tuple[Figure, list[Axes], list[DataFrame], Figure, list[Axes], list[DataFrame]]:

    fig_dis, axes_dis, df_list_dis = plot_comparison_label_distribution(partitioner_list=list(partitioner_dict.values()),label_name=label_name,plot_type="heatmap", size_unit= size_unit,
                                       max_num_partitions=max_num_partitions, partition_id_axis=partition_id_axis,figsize=figsize,
                                       subtitle="Comparison of Per Partition Label Distribution", titles=titles,cmap=cmap, legend=legend, legend_title=legend_title,
                                       verbose_labels=verbose_labels, plot_kwargs_list=plot_kwargs_list, legend_kwargs=legend_kwargs)
    if fairness == "attribute":
        fig, axes, df_list = plot_comparison_fairness_distribution(partitioner_dict=partitioner_dict,
                                                                   label_name=label_name, size_unit="absolute",
                                                                   max_num_partitions=max_num_partitions,
                                                                   partition_id_axis=partition_id_axis, figsize=figsize,
                                                                   subtitle=subtitle, titles=titles, cmap=cmap,
                                                                   legend=legend, legend_title=legend_title,
                                                                   verbose_labels=verbose_labels,
                                                                   plot_kwargs_list=plot_kwargs_list,
                                                                   legend_kwargs=legend_kwargs,
                                                                   fairness_metric=fairness_metric,
                                                                   model=model, class_label=class_label)
    elif fairness == "value":
        fig, axes, df_list = plot_comparison_fairness_distribution(partitioner_dict=partitioner_dict,
                                                                   label_name=label_name, size_unit="value",
                                                                   max_num_partitions=max_num_partitions,
                                                                   partition_id_axis=partition_id_axis, figsize=figsize,
                                                                   subtitle=subtitle, titles=titles, cmap=cmap,
                                                                   legend=legend, legend_title=legend_title,
                                                                   verbose_labels=verbose_labels,
                                                                   plot_kwargs_list=plot_kwargs_list,
                                                                   legend_kwargs=legend_kwargs,
                                                                   fairness_metric=fairness_metric,
                                                                   model=model,class_label=class_label)



    elif fairness == "attribute-value":
        fig, axes, df_list = plot_comparison_fairness_distribution(partitioner_dict=partitioner_dict,
                                                                   label_name=label_name, size_unit="attribute-value",
                                                                   max_num_partitions=max_num_partitions,
                                                                   partition_id_axis=partition_id_axis, figsize=figsize,
                                                                   subtitle=subtitle, titles=titles, cmap=cmap,
                                                                   legend=legend, legend_title=legend_title,
                                                                   verbose_labels=verbose_labels,
                                                                   plot_kwargs_list=plot_kwargs_list,
                                                                   legend_kwargs=legend_kwargs,
                                                                   fairness_metric=fairness_metric,
                                                                   model=model,class_label=class_label)
    return fig_dis,axes_dis, df_list_dis, fig, axes, df_list


def individual_fairness_plot(fairness_df_before, fairness_df_after):
    #this is for the evaluation after training the individuals FL algorithms
    pass

