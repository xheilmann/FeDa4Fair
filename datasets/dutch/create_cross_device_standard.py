"""
Creation script for Cross-Device Standard Unfairness Benchmarking Datasets.
Dataset: lucacorbucci/Dutch_census_binary_marital_status
Scenario: Cross-Device (150 clients)
All clients biased toward the same sensitive value.
Target DP Levels: Mild (0.15), Medium (0.25), Strong (0.35)
"""

from sklearn.linear_model import LogisticRegression

from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.metrics.fairness import compute_multi_fairness
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias


def create_benchmarks():
    num_clients = 150
    output_base = "datasets/dutch/cross_device_standard"

    levels = {
        "mild": {
            "drop_mean": 0.4, "flip_mean": 0.15, "mitigate_base": True, "target": 0.15
        },
        "medium": {
            "drop_mean": 0.3, "flip_mean": 0.1, "mitigate_base": False, "target": 0.25
        },
        "strong": {
            "drop_mean": 0.8, "flip_mean": 0.3, "mitigate_base": False, "target": 0.35
        }
    }

    for level_name, config in levels.items():
        print(f"Creating {level_name} benchmark (Target DP ~{config['target']})...")

        group_configs = [
            {
                "group_id": level_name,
                "num_clients": num_clients,
                "configs": [
                    {
                        "attribute": "sex_binary",
                        "value": 1,
                        "drop_mean": config["drop_mean"], "drop_std": 0.05,
                        "flip_mean": config["flip_mean"], "flip_std": 0.02,
                        "mitigate": config["mitigate_base"]
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
            fl_setting="cross-device",
            perc_train_val_test=[0.8, 0.2],
            path=f"{output_base}/{level_name}"
        )

        fds.prepare()

        # Evaluation
        print(f"Evaluating {level_name} benchmark...")
        results = compute_multi_fairness(
            partitioner=fds.partitioners["train"],
            partitioner_test=fds.partitioners["train"],
            model=LogisticRegression(max_iter=1000, solver="liblinear"),
            sens_atts=["sex_binary"],
            fairness_metric="DP",
            label_name="occupation_binary",
            fds=fds,
            split="train"
        )

        results_eo = compute_multi_fairness(
            partitioner=fds.partitioners["train"],
            partitioner_test=fds.partitioners["train"],
            model=LogisticRegression(max_iter=1000, solver="liblinear"),
            sens_atts=["sex_binary"],
            fairness_metric="EO",
            label_name="occupation_binary",
            fds=fds,
            split="train"
        )

        avg_dp = results["sex_binary_DP"].mean()
        avg_eo = results_eo["sex_binary_EO"].mean()
        print(f"Results for {level_name}: Avg DP={avg_dp:.4f}, Avg EO={avg_eo:.4f}")

        eval_path = f"{output_base}/{level_name}_evaluation.csv"
        results["sex_binary_EO"] = results_eo["sex_binary_EO"]
        results.to_csv(eval_path)
        print(f"Evaluation saved to {eval_path}\n")

if __name__ == "__main__":
    create_benchmarks()
