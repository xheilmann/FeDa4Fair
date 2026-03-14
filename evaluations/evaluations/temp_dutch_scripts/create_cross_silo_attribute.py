"""
Creation script for Cross-Silo Attribute Imbalanced Benchmarking Datasets.
Dataset: lucacorbucci/Dutch_census_binary_marital_status
Scenario: Cross-Silo (50 clients)
Target DP Levels: Medium (0.30)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias
from FeDa4Fair.visualization.plots import plot_multi_attribute_fairness


def create_benchmarks():
    num_clients = 50
    output_base = Path("datasets/dutch/cross_silo_attribute")

    if not output_base.exists():
        output_base.mkdir(parents=True)

    levels = {
        "medium": {
            "drop_mean": 0.7,
            "drop_std": 0.05,
            "flip_mean_sex": 0.4,
            "flip_mean_mar": 0.4,
            "flip_std": 0.02,
            "target": 0.30,
        }
    }

    for level_name, config in levels.items():
        print(f"Creating {level_name} benchmark (Target DP ~{config['target']})...")

        half_clients = num_clients // 2

        # Mitigate First, then Bias. Targeting value=1 for bias injection.
        group_configs = [
            {
                "group_id": "sex_bias",
                "num_clients": half_clients,
                "configs": [
                    {"attribute": "Marital_status", "mitigate": True},
                    {
                        "attribute": "sex_binary",
                        "value": 1,
                        "drop_mean": config["drop_mean"],
                        "drop_std": config["drop_std"],
                        "flip_mean": config["flip_mean_sex"],
                        "flip_std": config["flip_std"],
                        "mitigate": False,
                    },
                ],
            },
            {
                "group_id": "marital_bias",
                "num_clients": num_clients - half_clients,
                "configs": [
                    {"attribute": "sex_binary", "mitigate": True},
                    {
                        "attribute": "Marital_status",
                        "value": 1,
                        "drop_mean": config["drop_mean"],
                        "drop_std": config["drop_std"],
                        "flip_mean": config["flip_mean_mar"],
                        "flip_std": config["flip_std"],
                        "mitigate": False,
                    },
                ],
            },
        ]

        mod_dict = generate_multiobjective_bias(num_clients, group_configs)

        fds = FairFederatedDataset(
            dataset="lucacorbucci/Dutch_census_binary_marital_status",
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

        # Evaluation
        print(f"Evaluating {level_name} benchmark...")

        sens_atts = ["sex_binary", "Marital_status"]

        train_key = "train_train"
        if train_key not in fds.partitioners:
            train_key = next(iter(fds.partitioners.keys()))

        # Plot and compute DP
        fig_dp, _ax_dp, results_dp = plot_multi_attribute_fairness(
            partitioner=fds.partitioners[train_key],
            partitioner_test=fds.partitioners[train_key],
            model=LogisticRegression(max_iter=1000, solver="liblinear"),
            sens_atts=sens_atts,
            fairness_metric="DP",
            label_name="occupation_binary",
            fds=fds,
            split="train_train",
            figsize=(12, 6),
            title=f"Demographic Parity Distribution ({level_name})",
        )
        fig_dp.savefig(f"{output_base}/{level_name}_DP.png")
        plt.close(fig_dp)

        # Plot Accuracy
        if "Accuracy" in results_dp.columns:
            fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
            results_dp["Accuracy"].plot(kind="bar", ax=ax_acc, color="green")
            ax_acc.set_title(f"Local Model Accuracy ({level_name})")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_xlabel("Partition ID")
            fig_acc.savefig(f"{output_base}/{level_name}_Accuracy.png")
            plt.close(fig_acc)

        # Plot and compute EO
        fig_eo, _ax_eo, results_eo = plot_multi_attribute_fairness(
            partitioner=fds.partitioners[train_key],
            partitioner_test=fds.partitioners[train_key],
            model=LogisticRegression(max_iter=1000, solver="liblinear"),
            sens_atts=sens_atts,
            fairness_metric="EO",
            label_name="occupation_binary",
            fds=fds,
            split="train_train",
            figsize=(12, 6),
            title=f"Equalized Odds Distribution ({level_name})",
        )
        fig_eo.savefig(f"{output_base}/{level_name}_EO.png")
        plt.close(fig_eo)

        print(f"Results for {level_name}:")
        results = results_dp.copy()
        for att in sens_atts:
            avg_dp = results[f"{att}_DP"].mean()
            avg_eo = results_eo[f"{att}_EO"].mean()
            print(f"  {att}: Avg DP={avg_dp:.4f}, Avg EO={avg_eo:.4f}")
            results[f"{att}_EO"] = results_eo[f"{att}_EO"]

        eval_path = f"{output_base}/{level_name}_evaluation.csv"
        results.to_csv(eval_path)
        print(f"Evaluation saved to {eval_path}\n")


if __name__ == "__main__":
    create_benchmarks()
