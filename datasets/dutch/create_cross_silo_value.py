"""
Creation script for Cross-Silo Value Imbalanced Benchmarking Datasets.
Dataset: lucacorbucci/Dutch_census_binary_marital_status
Scenario: Cross-Silo (50 clients)
Target DP Levels: Medium (0.30)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.metrics.fairness import compute_multi_fairness
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias
from FeDa4Fair.visualization.plots import plot_multi_attribute_fairness


def evaluate_level(fds, level_name, output_base):
    # Evaluation
    print(f"Evaluating {level_name} benchmark...")

    sens_atts = ["sex_binary", "Marital_status"]
    train_key = "train_train"
    if train_key not in fds.partitioners:
        train_key = next(iter(fds.partitioners.keys()))

    # Calculate DATA Bias (model=None) to confirm generation correctness
    results_dp = compute_multi_fairness(
        partitioner=fds.partitioners[train_key],
        partitioner_test=fds.partitioners[train_key],
        model=None,  # Evaluate Data Bias!
        sens_atts=sens_atts,
        fairness_metric="DP",
        label_name="occupation_binary",
        fds=fds,
        split="train_train",
        size_unit="attribute-value",
    )

    # Plot Data Bias for each attribute
    for att in sens_atts:
        fig_dp, ax_dp = plt.subplots(figsize=(14, 6))
        cols = results_dp.columns

        # Find columns for bias toward values
        c_toward_0 = next((c for c in cols if c.startswith(f"{att}_") and ("_0.0_1.0" in c or "_0_1" in c)), None)
        c_toward_1 = next((c for c in cols if c.startswith(f"{att}_") and ("_1.0_0.0" in c or "_1_0" in c)), None)

        if c_toward_0 and c_toward_1:
            df_plot = pd.DataFrame(
                {
                    "Bias Toward 0 (Red)": results_dp[c_toward_0].clip(lower=0),
                    "Bias Toward 1 (Blue)": results_dp[c_toward_1].clip(lower=0),
                },
                index=results_dp.index,
            )

            df_plot.plot(kind="bar", ax=ax_dp, color=["red", "blue"], width=0.8)
            ax_dp.set_title(f"Data Demographic Parity Distribution: {att} ({level_name})")
            ax_dp.set_ylabel("DP Difference (Data Bias)")
            ax_dp.set_xlabel("Partition ID")
            ax_dp.grid(axis="y", linestyle="--", alpha=0.7)
            fig_dp.savefig(f"{output_base}/{level_name}_{att}_DP.png")
            
            # Print stats
            max_dp = results_dp[[c_toward_0, c_toward_1]].max(axis=1)
            print(f"  {att} (Data Bias): Avg DP={max_dp.mean():.4f}")
        else:
            print(f"Warning: Could not find DP columns for {att}. Cols: {cols}")
        plt.close(fig_dp)

    # Plot Accuracy (Need to train model now)
    results_model = compute_multi_fairness(
        partitioner=fds.partitioners[train_key],
        partitioner_test=fds.partitioners[train_key],
        model=LogisticRegression(max_iter=1000, solver="liblinear"),
        sens_atts=sens_atts,
        fairness_metric="DP",
        label_name="occupation_binary",
        fds=fds,
        split="train_train",
        size_unit="attribute",  # Standard summary
    )

    if "Accuracy" in results_model.columns:
        fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
        results_model["Accuracy"].plot(kind="bar", ax=ax_acc, color="green")
        ax_acc.set_title(f"Local Model Accuracy ({level_name})")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Partition ID")
        fig_acc.savefig(f"{output_base}/{level_name}_Accuracy.png")
        plt.close(fig_acc)

    # compute EO (Model)
    fig_eo, _, results_eo = plot_multi_attribute_fairness(
        partitioner=fds.partitioners[train_key],
        partitioner_test=fds.partitioners[train_key],
        model=LogisticRegression(max_iter=1000, solver="liblinear"),
        sens_atts=sens_atts,
        fairness_metric="EO",
        label_name="occupation_binary",
        fds=fds,
        split="train_train",
        size_unit="value",
        value_colors={0.0: "red", 1.0: "blue"},
    )
    fig_eo.savefig(f"{output_base}/{level_name}_EO.png")
    plt.close(fig_eo)

    # Save merged results
    results = results_dp.copy()
    results["Accuracy"] = results_model["Accuracy"]

    for col in results_eo.columns:
        if col not in results.columns:
            results[col] = results_eo[col]

    eval_path = f"{output_base}/{level_name}_evaluation.csv"
    results.to_csv(eval_path)
    print(f"Evaluation saved to {eval_path}\n")


def create_benchmarks():
    num_clients = 50
    output_base = Path("datasets/dutch/cross_silo_value")

    output_base.mkdir(parents=True, exist_ok=True)

    # Tuning params:
    levels = {
        "medium": {
            "drop_mean": 0.5,
            "drop_std": 0.05,
            "flip_mean_0": 0.4,
            "flip_mean_1": 0.4,
            "flip_std": 0.02,
            "target": 0.30,
        }
    }

    for level_name, config in levels.items():
        print(f"Creating {level_name} benchmark (Target DP ~{config['target']})...")

        half_clients = num_clients // 2

        group_configs = [
            {
                "group_id": "value_0_bias",
                "num_clients": half_clients,
                "configs": [
                    {
                        "attribute": "sex_binary",
                        "value": 0,
                        "drop_mean": config["drop_mean"],
                        "drop_std": config["drop_std"],
                        "flip_mean": config["flip_mean_0"],
                        "flip_std": config["flip_std"],
                        "mitigate": False,
                    }
                ],
            },
            {
                "group_id": "value_1_bias",
                "num_clients": num_clients - half_clients,
                "configs": [
                    {
                        "attribute": "sex_binary",
                        "value": 1,
                        "drop_mean": config["drop_mean"],
                        "drop_std": config["drop_std"],
                        "flip_mean": config["flip_mean_1"],
                        "flip_std": config["flip_std"],
                        "mitigate": False,
                    }
                ],
            },
        ]

        mod_dict = generate_multiobjective_bias(num_clients, group_configs)

        fds = FairFederatedDataset(
            dataset="lucacorbucci/Dutch_census",
            split="all",
            partitioners={"train": num_clients},
            label_name="occupation_binary",
            sensitive_attributes=["sex_binary", "Marital_status"],
            modification_dict=mod_dict,
            fl_setting="cross-silo",
            perc_train_val_test=[0.8, 0.2],
            path=f"{output_base}/{level_name}",
        )

        fds.prepare()
        evaluate_level(fds, level_name, output_base)


if __name__ == "__main__":
    create_benchmarks()
