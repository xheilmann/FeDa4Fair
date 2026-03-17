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

"""Evaluation functions for fairness."""

import pickle
from pathlib import Path
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flwr_datasets.partitioner import Partitioner
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from FeDa4Fair.metrics.fairness import _compute_fairness
from FeDa4Fair.utils.constants import (
    DEFAULT_SENSITIVE_ATTRIBUTES,
    LEVEL_ATTRIBUTE,
    LEVEL_VALUE,
)
from FeDa4Fair.visualization.plots import (
    plot_comparison_fairness_distribution,
    plot_comparison_label_distribution,
)

# Try to import XGBoost, but make it optional
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier: Any = None
    XGBOOST_AVAILABLE = False
except Exception:  # noqa: BLE001
    XGBClassifier: Any = None
    XGBOOST_AVAILABLE = False


def evaluate_fairness(
    partitioner_dict: dict[str, Partitioner],
    max_num_partitions: int | None = 10,
    sens_columns: str | list[str] | None = None,
    intersectional_fairness: list[str] | None = None,
    size_unit: Literal["percent", "absolute"] = "absolute",
    fairness_metric: Literal["DP", "EO"] = "DP",
    fairness_level: Literal["attribute", "value", "attribute-value"] = "attribute",
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: tuple[float, float] | None = None,
    subtitle: str = "Fairness Distribution Per Partition",
    titles: list[str] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    verbose_labels: bool = False,
    plot_kwargs_list: list[dict[str, Any] | None] | None = None,
    legend_kwargs: dict[Any, Any] | None = None,
    model: Any | None = None,
    label_name: str = "label",
    path: str = "data_stats",
) -> None:
    """
    Save, evaluate and visualize fairness metrics and data counts across partitions.

    Parameters
    ----------
    partitioner_dict : dict[str, Partitioner]
        A dictionary where keys are labels or dataset identifiers, and values are Partitioner objects.

    max_num_partitions : int, optional
        The maximum number of partitions to display per dataset (default is 10).

    sens_columns : str or list of str, optional
        Sensitive attribute(s) used to evaluate fairness. Defaults to ["SEX", "MAR", "RAC1P"] if None.

    intersectional_fairness : list[str], optional
        If provided, evaluate intersectional fairness using combinations of the listed attributes.

    size_unit : {"percent", "absolute"}, default="absolute"
        Whether to express data counts as percentages or absolute counts.

    fairness_metric : {"DP", "EO"}, default="DP"
        Fairness metric to evaluate. "DP" = Demographic Parity, "EO" = Equalized Odds.

    fairness_level : {"attribute", "value", "attribute-value"}, default="attribute"
        The level at which fairness is evaluated.

    partition_id_axis : {"x", "y"}, default="y"
        Axis to use for partition labels.

    figsize : tuple(float, float), optional
        Custom figure size.

    subtitle : str, default="Fairness Distribution Per Partition"
        Subtitle to display.

    titles : list[str], optional
        A list of titles, one for each subplot.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for visualization.

    legend : bool, default=False
        Whether to display a legend.

    legend_title : str, optional
        Title for the legend.

    verbose_labels : bool, default=False
        Whether to show detailed labels.

    plot_kwargs_list : list of dict, optional
        Additional keyword arguments for plotting.

    legend_kwargs : dict, optional
        Additional keyword arguments for legend.

    model : optional
        Model object to use for evaluation.

    label_name : str, default="label"
        The name of the label column.

    path : str, default="data_stats"
        Output path for saved files.

    Returns
    -------
    None
    """
    if sens_columns is None:
        sens_columns = DEFAULT_SENSITIVE_ATTRIBUTES

    if isinstance(sens_columns, str):
        sens_columns = [sens_columns]

    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    for label in sens_columns:
        fig_dis, axes_dis, df_list_dis = plot_comparison_label_distribution(
            partitioner_list=list(partitioner_dict.values()),
            label_name=label,
            plot_type="heatmap",
            size_unit=size_unit,
            max_num_partitions=max_num_partitions,
            partition_id_axis=partition_id_axis,
            figsize=figsize,
            subtitle="Comparison of Per Partition Label Distribution",
            titles=titles,
            cmap=cmap,
            legend=legend,
            legend_title=legend_title,
            verbose_labels=verbose_labels,
            plot_kwargs_list=plot_kwargs_list,
            legend_kwargs=legend_kwargs,
        )

        merged_df = merge_dataframes_with_names(df_list_dis, list(partitioner_dict.keys()))
        merged_df.to_csv(output_path / f"{label}_count_df.csv")
        fig_dis.savefig(output_path / f"{label}_count_fig.pdf", dpi=1200)

        with (output_path / "fig_ax_count.pkl").open("wb") as f:
            pickle.dump({"fig": fig_dis, "ax": axes_dis}, f)

    names = list(partitioner_dict.keys())
    if model is not None:
        names = [key for key in partitioner_dict if "train" in key]

    eff_sens_columns = sens_columns
    if intersectional_fairness is not None:
        eff_sens_columns = ["_".join(intersectional_fairness)]

    for sens_att in eff_sens_columns:
        fig, axes, df_list = plot_comparison_fairness_distribution(
            partitioner_dict=partitioner_dict,
            sens_att=sens_att,
            sens_cols=sens_columns,
            size_unit=fairness_level,
            max_num_partitions=max_num_partitions,
            partition_id_axis=partition_id_axis,
            figsize=figsize,
            subtitle=subtitle,
            titles=titles,
            cmap=cmap,
            legend=legend,
            plot_kwargs_list=plot_kwargs_list,
            legend_kwargs=legend_kwargs,
            fairness_metric=fairness_metric,
            model=model,
            label_name=label_name,
            intersectional_fairness=intersectional_fairness,
        )
        df_fairness = merge_dataframes_with_names(df_list, names)
        df_fairness.to_csv(output_path / f"{sens_att}_{fairness_metric}_df.csv")
        fig.savefig(output_path / f"{sens_att}_{fairness_metric}_fig.pdf", dpi=1200)

        with (output_path / f"fig_ax_{fairness_metric}.pkl").open("wb") as f:
            pickle.dump({"fig": fig, "ax": axes}, f)


def local_client_fairness_plot(
    before_fairness_df: pd.DataFrame,
    after_fairness_df: pd.DataFrame,
    client_column: str = "Partition ID",
    fairness_column: str = "RAC1P_DP",
    title: str = "Fairness Before/After Comparison",
    figsize: tuple = (6, 6),
    ylabel: str = "Fairness Value Before",
    xlabel: str = "Fairness Value After",
) -> plt.Figure:
    """
    Plot a scatter comparison of fairness values from two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame.
    df2 : pd.DataFrame
        Second DataFrame.
    client_column : str, default="Partition ID"
        Name of client ID column.
    fairness_column : str, default="RAC1P_DP"
        Name of fairness metric column.
    title : str, optional
        Title of plot.
    figsize : tuple, optional
        Figure size.
    ylabel: str, optional
        Y-axis label.
    xlabel: str, optional
        X-axis label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not before_fairness_df[client_column].is_unique:
        msg = "The client ID column must be unique for before_fairness_df."
        raise ValueError(msg)

    merged = (
        before_fairness_df[[client_column, fairness_column]]
        .rename(columns={fairness_column: "fairness1"})
        .merge(
            after_fairness_df[[client_column, fairness_column]].rename(columns={fairness_column: "fairness2"}),
            on=client_column,
        )
    )

    fairness1 = merged["fairness1"]
    fairness2 = merged["fairness2"]

    min_val = min(fairness1.min(), fairness2.min())
    max_val = max(fairness1.max(), fairness2.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(fairness2, fairness1, alpha=0.7)
    ax.plot([min_val - 0.05, max_val + 0.05], [min_val - 0.05, max_val + 0.05], linestyle="dotted", color="gray")

    ax.set_xlim(min_val - 0.05, max_val + 0.05)
    ax.set_ylim(min_val - 0.05, max_val + 0.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(visible=True)

    return fig


def _get_models() -> dict[str, Any]:
    """Return fresh, unfitted model instances for each evaluation."""
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=5000),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss")  # type: ignore[no-redef]
    return models


def evaluate_model(
    model_name: str,
    model: Any,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    fairness_metric: Literal["DP", "EO"],
    sf_data: dict[str, np.ndarray],
    fairness_level: str,
) -> dict:
    """Trains and evaluates a classification model."""
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    results = {"model": model_name, "accuracy": acc}
    for key, value in sf_data.items():
        sf_df = pd.DataFrame(value, columns=[key])  # type: ignore[arg-type]
        if fairness_level == LEVEL_VALUE:
            results[f"value_{fairness_metric}_{key}"] = _compute_fairness(
                y_test, preds, sf_df, fairness_metric, key, LEVEL_VALUE
            ).to_numpy()[1]

        results[f"{fairness_metric}_{key}"] = _compute_fairness(
            y_test, preds, sf_df, fairness_metric, key, LEVEL_ATTRIBUTE
        ).to_numpy()[0]

    return results


def evaluate_models_on_datasets(
    datasets: list, n_jobs: int = -1, fairness_metric: str = "DP", fairness_level: str = "attribute"
) -> tuple[pd.DataFrame, Any]:
    """Evaluates multiple models on multiple datasets in parallel."""
    tasks = []
    models = _get_models()

    for entry in datasets:
        # Support both tuple and ProcessedSiloData
        if hasattr(entry, "to_tuple"):
            name, x_train, y_train, x_test, y_test, sf_data = entry.to_tuple()
        else:
            name, x_train, y_train, x_test, y_test, sf_data = entry

        for model_name, model in models.items():
            tasks.append(
                delayed(evaluate_model)(
                    model_name,
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    fairness_metric=fairness_metric,
                    sf_data=sf_data,
                    fairness_level=fairness_level,
                )
            )

    results = Parallel(n_jobs=n_jobs)(tasks)

    expanded_results = []
    for i, res in enumerate(results):
        dataset_index = i // len(models)
        entry = datasets[dataset_index]
        dataset_name = entry.name if hasattr(entry, "name") else entry[0]
        res["dataset"] = dataset_name
        expanded_results.append(res)

    results_df = pd.DataFrame(expanded_results)
    # The plotting logic is removed and should be called separately from visualization.plots
    return results_df, None


def merge_dataframes_with_names(dfs: list[pd.DataFrame], names: list[str], name_column: str = "state") -> pd.DataFrame:
    """Merges a list of DataFrames and adds a column indicating their source."""
    if len(dfs) != len(names):
        msg = "Each DataFrame must have a corresponding name."
        raise ValueError(msg)

    tagged_dfs = []
    for df, name in zip(dfs, names, strict=False):
        df_copy = df.copy()
        df_copy[name_column] = name
        tagged_dfs.append(df_copy)

    return pd.concat(tagged_dfs)
