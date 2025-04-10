import os
import pickle
from itertools import product

import pandas as pd

from comparision_count import plot_comparison_label_distribution
from pandas import DataFrame

from comparision_fairness_distribution import plot_comparison_fairness_distribution
from typing import Any, Optional, Union, Literal

import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flwr_datasets.partitioner import Partitioner



def evaluate_fairness(
    partitioner_dict: dict[str, Partitioner],
    max_num_partitions: Optional[int] = 30,
    label_name: Union[str, list[str]]= ["SEX", "MAR", "RAC1P"],
    intersectional_fairness: list[str] = None,
    size_unit: Literal["percent", "absolute"] = "absolute",
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
    path: str = "data_stats",
) -> None:

    for label in label_name:
        fig_dis, axes_dis, df_list_dis = plot_comparison_label_distribution(partitioner_list=list(partitioner_dict.values()),label_name=label,plot_type="heatmap", size_unit= size_unit,
                                           max_num_partitions=max_num_partitions, partition_id_axis=partition_id_axis,figsize=figsize,
                                           subtitle="Comparison of Per Partition Label Distribution", titles=titles,cmap=cmap, legend=legend, legend_title=legend_title,
                                           verbose_labels=verbose_labels, plot_kwargs_list=plot_kwargs_list, legend_kwargs=legend_kwargs)

        df = merge_dataframes_with_names(df_list_dis, list(partitioner_dict.keys()))
        df.to_csv(os.path.join(path, f"{label}_count_df.csv" ))
        fig_dis.savefig(os.path.join(path, f"{label}_count_fig.pdf"), dpi=1200)

        with open(f"{path}/fig_ax_count.pkl", 'wb') as f:
            pickle.dump({'fig': fig_dis, 'ax': axes_dis}, f)

    if intersectional_fairness is not None:
        label_name = [f"{intersectional_fairness}"]

    for label in label_name:
        fig, axes, df_list = plot_comparison_fairness_distribution(partitioner_dict=partitioner_dict,
                                                                   label_name=label, size_unit=fairness,
                                                                   max_num_partitions=max_num_partitions,
                                                                   partition_id_axis=partition_id_axis, figsize=figsize,
                                                                   subtitle=subtitle, titles=titles, cmap=cmap,
                                                                   legend=legend, legend_title=legend_title,
                                                                   verbose_labels=verbose_labels,
                                                                   plot_kwargs_list=plot_kwargs_list,
                                                                   legend_kwargs=legend_kwargs,
                                                                   fairness_metric=fairness_metric,
                                                                   model=model, class_label=class_label, intersectional_fairness = intersectional_fairness)
        fig.show()
        df_fairness = merge_dataframes_with_names(df_list, list(partitioner_dict.keys()))
        df_fairness.to_csv(os.path.join(path, f"{label}_{fairness_metric}_df.csv"))
        fig.savefig(os.path.join(path, f"{label}_{fairness_metric}_fig.pdf"), dpi=1200)

        with open(f"{path}/fig_ax_{fairness_metric}.pkl", 'wb') as f:
            pickle.dump({'fig': fig, 'ax': axes}, f)
        #print(df_list)



def local_client_fairness_plot(fairness_df_before, fairness_df_after):
    #TODO this is for the evaluation after training the individual FL algorithms
    pass



def merge_dataframes_with_names(dfs, names, name_column='state'):
    """
    Merges a list of dataframes and adds a column indicating their source.

    :param dfs: List of pandas DataFrames.
    :param names: List of names (same length as dfs).
    :param name_column: Name of the new column that tags the source.
    :return: Merged DataFrame.
    """
    assert len(dfs) == len(names), "Each DataFrame must have a corresponding name."

    tagged_dfs = []
    for df, name in zip(dfs, names):
        df_copy = df.copy()
        df_copy[name_column] = name
        tagged_dfs.append(df_copy)

    return pd.concat(tagged_dfs)