"""
Creation script for Cross-Silo Value Imbalanced Benchmarking Datasets.
Dataset: lucacorbucci/Dutch_census_binary_marital_status
Scenario: Cross-Silo (50 clients)
Target DP Levels: Mild (0.15), Medium (0.25), Strong (0.35)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import selection_rate, MetricFrame

from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias
from FeDa4Fair.visualization.plots import plot_multi_attribute_fairness
from FeDa4Fair.metrics.fairness import compute_multi_fairness

def add_proxies(dataset_dict):
    """Add proxy columns so models can learn bias when sensitive attributes are dropped."""
    for split in dataset_dict.keys():
        dataset_dict[split] = dataset_dict[split].map(
            lambda x: {"Sex_Proxy": x["sex_binary"], "Marital_Proxy": x["Marital_status"]},
            batched=False
        )
    return dataset_dict

def create_benchmarks():
    num_clients = 50
    output_base = "datasets/dutch/cross_silo_value"
    
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    levels = {
        "mild": {
            "drop_mean": 0.2, "drop_std": 0.05,
            "flip_mean_0": 0.6, "flip_mean_1": 0.2, "flip_std": 0.02,
            "target": 0.15
        },
        "medium": {
            "drop_mean": 0.2, "drop_std": 0.05,
            "flip_mean_0": 0.75, "flip_mean_1": 0.4, "flip_std": 0.02,
            "target": 0.25
        },
        "strong": {
            "drop_mean": 0.2, "drop_std": 0.05,
            "flip_mean_0": 0.9, "flip_mean_1": 0.6, "flip_std": 0.02,
            "target": 0.35
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
                        "drop_mean": config["drop_mean"], "drop_std": config["drop_std"],
                        "flip_mean": config["flip_mean_0"], "flip_std": config["flip_std"],
                        "mitigate": False
                    }
                ]
            },
            {
                "group_id": "value_1_bias",
                "num_clients": num_clients - half_clients,
                "configs": [
                    {
                        "attribute": "sex_binary",
                        "value": 1,
                        "drop_mean": config["drop_mean"], "drop_std": config["drop_std"],
                        "flip_mean": config["flip_mean_1"], "flip_std": config["flip_std"],
                        "mitigate": False
                    }
                ]
            }
        ]

        mod_dict = generate_multiobjective_bias(num_clients, group_configs)

        fds = FairFederatedDataset(
            dataset="lucacorbucci/Dutch_census_binary_marital_status",
            split="all",
            partitioners={"train": num_clients},
            label_name="occupation_binary",
            sensitive_attributes=["sex_binary"],
            modification_dict=mod_dict,
            fl_setting="cross-silo",
            perc_train_val_test=[0.8, 0.2],
            path=f"{output_base}/{level_name}",
            preprocessor=add_proxies
        )

        fds.prepare()

        # Evaluation
        print(f"Evaluating {level_name} benchmark...")
        
        sens_atts = ["sex_binary"]
        train_key = "train_train"
        if train_key not in fds.partitioners:
            train_key = list(fds.partitioners.keys())[0]

        # Manual loop to collect Selection Rates and Accuracy
        partition_stats = []
        
        num_parts = fds.partitioners[train_key].num_partitions
        for pid in range(num_parts):
            # Load
            partition = fds.load_partition(pid, split="train_train")
            df = partition.to_pandas()
            
            # Train Model
            # Drop sensitive attributes (but keep proxies)
            cols_to_drop = sens_atts + ["occupation_binary"]
            X = df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include=["number", "bool"])
            y = df["occupation_binary"]
            
            model = LogisticRegression(max_iter=1000, solver="liblinear")
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate Accuracy
            acc = accuracy_score(y, y_pred)
            
            # Calculate Selection Rates for Sex
            # We need the sensitive attribute column, which we dropped from X but is in df
            sex_col = df["sex_binary"]
            mf = MetricFrame(metrics=selection_rate, y_true=y, y_pred=y_pred, sensitive_features=sex_col)
            sr_by_group = mf.by_group
            
            partition_stats.append({
                "Partition ID": pid,
                "Accuracy": acc,
                "SR_0": sr_by_group.get(0, 0),
                "SR_1": sr_by_group.get(1, 0),
                "DP": abs(sr_by_group.get(0, 0) - sr_by_group.get(1, 0))
            })

        stats_df = pd.DataFrame(partition_stats).set_index("Partition ID")

        # Plot Selection Rates (Two bars per client)
        fig_sr, ax_sr = plt.subplots(figsize=(14, 6))
        stats_df[["SR_0", "SR_1"]].plot(kind="bar", ax=ax_sr, color=["red", "blue"], width=0.8)
        ax_sr.set_title(f"Selection Rates by Group ({level_name})")
        ax_sr.set_ylabel("Selection Rate")
        ax_sr.set_xlabel("Partition ID")
        ax_sr.legend(["Group 0 (Female)", "Group 1 (Male)"])
        ax_sr.grid(axis='y', linestyle='--', alpha=0.7)
        fig_sr.savefig(f"{output_base}/{level_name}_SelectionRates.png")
        plt.close(fig_sr)

        # Plot Accuracy
        fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
        stats_df["Accuracy"].plot(kind="bar", ax=ax_acc, color="green")
        ax_acc.set_title(f"Local Model Accuracy ({level_name})")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Partition ID")
        fig_acc.savefig(f"{output_base}/{level_name}_Accuracy.png")
        plt.close(fig_acc)

        # Print Avg DP
        avg_dp = stats_df["DP"].mean()
        print(f"  sex_binary: Avg DP={avg_dp:.4f}")

        # Save CSV
        eval_path = f"{output_base}/{level_name}_evaluation.csv"
        stats_df.to_csv(eval_path)
        print(f"Evaluation saved to {eval_path}\n")

if __name__ == "__main__":
    create_benchmarks()
