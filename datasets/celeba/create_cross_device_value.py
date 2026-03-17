"""
Creation script for Cross-Device Value Imbalanced Benchmarking Datasets for CelebA.
Dataset: flwrlabs/celeba
Scenario: Cross-Device (150 clients)
Target DP Level: Medium (0.30)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from datasets import concatenate_datasets, load_dataset
from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias
from FeDa4Fair.visualization.plots import plot_multi_attribute_fairness


def add_hair_color_multi(df):
    """
    Adds 'hair_color' column with multiple values.
    0: Black_Hair
    1: Blond_Hair
    2: Brown_Hair
    3: Gray_Hair
    4: Other
    """
    c_black = (df["Black_Hair"] == 1) | (df["Black_Hair"])
    c_blond = (df["Blond_Hair"] == 1) | (df["Blond_Hair"])
    c_brown = (df["Brown_Hair"] == 1) | (df["Brown_Hair"])
    c_gray = (df["Gray_Hair"] == 1) | (df["Gray_Hair"])

    conditions = [c_black, c_blond, c_brown, c_gray]
    choices = [0, 1, 2, 3]

    df["hair_color"] = np.select(conditions, choices, default=4)
    return df


def get_celeba_dataframe(img_dir_path):
    print("Loading CelebA dataset...")
    ds_dict = load_dataset("flwrlabs/celeba")
    ds_merged = concatenate_datasets(list(ds_dict.values()))

    print("Adding image IDs...")
    ds_merged = ds_merged.add_column("image_id", range(len(ds_merged)))

    # Check if image directory exists
    if not img_dir_path.exists():
        print(f"Saving individual images to {img_dir_path}...")
        img_dir_path.mkdir(parents=True, exist_ok=True)
        for item in ds_merged:
            idx = item["image_id"]
            img = item["image"]
            img.save(img_dir_path / f"{idx}.png")
        print("Images saved.")
    else:
        print(f"Image directory found at {img_dir_path}, skipping creation.")

    print("Dropping image column...")
    ds_merged = ds_merged.remove_columns("image")

    print("Converting to Pandas for preprocessing...")
    df = ds_merged.to_pandas()

    print("Adding 'hair_color' attribute (multi-value)...")
    return add_hair_color_multi(df)


def evaluate_benchmark(fds, output_base, level_name):
    print(f"Evaluating {level_name} benchmark...")

    sens_atts = ["hair_color"]
    train_key = "train_train" if "train_train" in fds.partitioners else "train"

    fig, _ax, results_dp_values = plot_multi_attribute_fairness(
        partitioner=fds.partitioners[train_key],
        partitioner_test=fds.partitioners[train_key],
        model=None,
        sens_atts=sens_atts,
        fairness_metric="DP",
        label_name="Smiling",
        fds=fds,
        split=train_key,
        size_unit="value",
    )
    plt.close(fig)

    hair_cols = [c for c in results_dp_values.columns if "hair_color" in c]
    max_dp_per_client = results_dp_values[hair_cols].max(axis=1)
    idxmax_per_client = results_dp_values[hair_cols].idxmax(axis=1)

    def get_color(col_name):
        if not isinstance(col_name, str):
            return "gray"
        if "_0_" in col_name or col_name.endswith("_0"):
            return "red"
        if "_1_" in col_name or col_name.endswith("_1"):
            return "blue"
        if "_2_" in col_name or col_name.endswith("_2"):
            return "green"
        if "_3_" in col_name or col_name.endswith("_3"):
            return "orange"
        return "gray"

    colors = idxmax_per_client.apply(get_color)

    fig_custom, ax_custom = plt.subplots(figsize=(16, 6))
    max_dp_per_client.plot(kind="bar", ax=ax_custom, color=colors)
    ax_custom.set_title(f"Max Unfairness (DP) per Client by Value ({level_name})")
    ax_custom.set_xlabel("Client ID")
    ax_custom.set_ylabel("Max DP across Hair Colors")

    n = len(max_dp_per_client)
    ax_custom.set_xticks(range(0, n, 10))
    ax_custom.set_xticklabels(range(0, n, 10))

    custom_lines = [
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="blue", lw=4),
        Line2D([0], [0], color="green", lw=4),
        Line2D([0], [0], color="orange", lw=4),
    ]
    ax_custom.legend(custom_lines, ["Black (0)", "Blond (1)", "Brown (2)", "Gray (3)"])

    fig_custom.savefig(f"{output_base}/{level_name}_MaxDP.png")
    plt.close(fig_custom)

    results_dp_values["Max_DP"] = max_dp_per_client
    eval_path = f"{output_base}/{level_name}_evaluation.csv"
    results_dp_values.to_csv(eval_path)
    print(f"Evaluation saved to {eval_path}\n")
    print(f"Average Max DP across clients: {max_dp_per_client.mean():.4f}")


def create_benchmarks():
    num_clients = 150
    output_base = "datasets/celeba/cross_device_value"
    img_dir_path = Path("datasets/celeba/images")

    df = get_celeba_dataframe(img_dir_path)

    counts = df["hair_color"].value_counts()
    print("Hair Color Counts:\n", counts)

    named_counts = counts[counts.index.isin([0, 1, 2, 3])]
    val_max = named_counts.idxmax()
    val_min = named_counts.idxmin()

    print(f"Targeting Max Present: {val_max} (Count: {named_counts[val_max]})")
    print(f"Targeting Min Present: {val_min} (Count: {named_counts[val_min]})")

    level_name = "medium"
    print(f"Creating {level_name} benchmark (Target DP ~0.30)...")

    half_clients = num_clients // 2

    # Tuned parameters: 0.35 flip
    config = {"drop_mean": 0.3, "drop_std": 0.05, "flip_mean": 0.35, "flip_std": 0.02, "target": 0.30}

    group_configs = [
        {
            "group_id": "unfair_min_value",
            "num_clients": half_clients,
            "configs": [
                {
                    "attribute": "hair_color",
                    "value": val_min,
                    "drop_mean": config["drop_mean"],
                    "drop_std": config["drop_std"],
                    "flip_mean": config["flip_mean"],
                    "flip_std": config["flip_std"],
                    "mitigate": False,
                }
            ],
        },
        {
            "group_id": "unfair_max_value",
            "num_clients": num_clients - half_clients,
            "configs": [
                {
                    "attribute": "hair_color",
                    "value": val_max,
                    "drop_mean": config["drop_mean"],
                    "drop_std": config["drop_std"],
                    "flip_mean": config["flip_mean"],
                    "flip_std": config["flip_std"],
                    "mitigate": False,
                }
            ],
        },
    ]

    mod_dict = generate_multiobjective_bias(num_clients, group_configs)

    fds = FairFederatedDataset(
        dataset="flwrlabs/celeba",
        preloaded_data=df,
        split="all",
        partitioners={"train": num_clients},
        label_name="Smiling",
        sensitive_attributes=["hair_color"],
        modification_dict=mod_dict,
        fl_setting="cross-device",
        perc_train_val_test=[0.8, 0.2],
        path=f"{output_base}/{level_name}",
    )

    fds.prepare()
    evaluate_benchmark(fds, output_base, level_name)


if __name__ == "__main__":
    create_benchmarks()
