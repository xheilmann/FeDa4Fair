import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt

from fairness_computation import _compute_fairness
from plots import plot_comparison_label_distribution, plot_comparison_fairness_distribution
from typing import Any, Optional, Union, Literal
import matplotlib.colors as mcolors
from flwr_datasets.partitioner import Partitioner
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import seaborn as sns



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


# Dictionary of models to evaluate
MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'MLP': MLPClassifier(max_iter=1000),
    'XGBoost': XGBClassifier( eval_metric='logloss')
}


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, fairness_metric, sf_data):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    dict = {'model': model_name,
        'accuracy': acc }
    for key, value in sf_data.items():
        dict[f"{fairness_metric}_{key}"] = _compute_fairness(y_test, preds, value, fairness_metric,key, "attribute").values[0]
    return dict


def evaluate_models_on_datasets(datasets, n_jobs=-1, fairness_metric= "EO"):
    """
    Evaluates multiple models on multiple datasets in parallel.

    Parameters:
    - datasets: list of tuples (name, X_train, y_train, X_test, y_test, sf_data) where _train, _test are numpy arrays and sf_data a dictionary of the form {"sensitive_attribute": np.array(sensitive_values),...}
    - n_jobs: number of parallel jobs (default -1 = all cores)
    - fairness_metric: string of metric to use for fairness, possible to choose from demographic pariety ("DP") and Equalized Odds ("EO") (default "DP")

    Returns:
    - Pandas DataFrame of results, figure for each dataset in parallel.
    """


    tasks = []

    for dataset_name, X_train, y_train, X_test, y_test, sf_data in datasets:
        for model_name, model in MODELS.items():
            tasks.append(delayed(evaluate_model)(
                model_name, model, X_train, y_train, X_test, y_test, fairness_metric=fairness_metric,sf_data=sf_data,
            ))

    results = Parallel(n_jobs=n_jobs)(tasks)

    expanded_results = []
    for i, res in enumerate(results):
        dataset_index = i // len(MODELS)
        dataset_name = datasets[dataset_index][0]
        res['dataset'] = dataset_name
        expanded_results.append(res)

    df = pd.DataFrame(expanded_results)

    fairness_columns = [col for col in df.columns if col.startswith(f'{fairness_metric}_')]
    models = df['model'].unique()

    for col in fairness_columns:
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        ax = sns.barplot(
            data=df,
            x='dataset',
            y=col,
            hue=df['model'],
            dodge=True
        )

        for i, row in df.iterrows():
            group_val = row['dataset']
            model = row['model']
            acc = row['accuracy']

            # Get the position of the corresponding bar
            bar_index = list(df['dataset'].unique()).index(group_val)
            hue_index = list(models).index(model)
            total_hues = len(models)
            bar_width = 0.8 / total_hues
            offset = (hue_index - (total_hues - 1) / 2) * bar_width

            x_pos = bar_index + offset
            ax.plot(x_pos, acc, marker='x', color='red', markersize=10, label='Accuracy' if i == 0 else "")

        # Add labels and legend
        ax.set_title(f"{col} / accuracy")
        ax.set_ylabel(f"{fairness_metric} / Accuracy")
        ax.set_xlabel("Client")
        handles, labels = ax.get_legend_handles_labels()
        if 'Accuracy' not in labels:
            handles.append(plt.Line2D([0], [0], marker='x', color='red', linestyle='', label='Accuracy'))
            labels.append('Accuracy')
        ax.legend(handles, labels)
        plt.tight_layout()
        plt.show()

    return df, plt


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