"""
Creation script for Cross-Device Standard Unfairness Benchmarking Datasets for CelebA.
Dataset: flwrlabs/celeba
Scenario: Cross-Device (150 clients)
All clients biased toward the same sensitive value.
Target DP Levels: Mild, Medium, Strong
"""

import pandas as pd
import numpy as np
from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias

def compute_dataset_dp(fds, split):
    dps = []
    # Determine number of clients
    partitioner = fds.partitioners[split]
    if isinstance(partitioner, int):
        num_clients = partitioner
    else:
        num_clients = partitioner.num_partitions

    print(f"Computing DP for {num_clients} clients...")
    
    for cid in range(num_clients):
        try:
            partition = fds.load_partition(cid, split)
            df = partition.to_pandas()
            
            # Sensitive: "Male", Label: "Smiling"
            # Assuming boolean or 0/1
            sens = df["Male"]
            label = df["Smiling"]
            
            # Cast to numeric if boolean for mean calculation, though mean works on bools too
            # Check for True/1 and False/0
            s1_mask = (sens == True) | (sens == 1) | (sens == "true") | (sens == "True")
            s0_mask = (sens == False) | (sens == 0) | (sens == "false") | (sens == "False")
            
            if s1_mask.sum() == 0 or s0_mask.sum() == 0:
                continue
                
            p_y1_s1 = label[s1_mask].mean()
            p_y1_s0 = label[s0_mask].mean()
            
            dp = abs(p_y1_s1 - p_y1_s0)
            dps.append(dp)
        except Exception as e:
            print(f"Error computing DP for client {cid}: {e}")
            continue
        
    if not dps:
        return 0.0
    return sum(dps) / len(dps)

def create_benchmarks():
    num_clients = 150
    output_base = "datasets/celeba/cross_device_standard"

    # Using same levels as Dutch for consistency
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

    results_summary = []

    for level_name, config in levels.items():
        print(f"Creating {level_name} benchmark (Target DP ~{config['target']})...")

        # Standard: One group, all clients same bias
        # Bias against Males (value=True) to reduce smiling males?
        # Or bias against Females?
        # In Dutch Standard: value=1 (which was Sex, 1=Male usually).
        # We'll target Male=True (1) here.
        group_configs = [
            {
                "group_id": level_name,
                "num_clients": num_clients,
                "configs": [
                    {
                        "attribute": "Male",
                        "value": True, # Target Males
                        "drop_mean": config["drop_mean"], "drop_std": 0.05,
                        "flip_mean": config["flip_mean"], "flip_std": 0.02,
                        "mitigate": config["mitigate_base"]
                    }
                ]
            }
        ]

        mod_dict = generate_multiobjective_bias(num_clients, group_configs)

        fds = FairFederatedDataset(
            dataset="flwrlabs/celeba",
            split="all",
            partitioners={"train": num_clients},
            label_name="Smiling",
            sensitive_attributes=["Male"],
            modification_dict=mod_dict,
            fl_setting="cross-device",
            perc_train_val_test=[0.8, 0.2],
            path=f"{output_base}/{level_name}"
        )

        # Prepare (downloads if needed, partitions, applies mods)
        fds.prepare()

        # Evaluation
        print(f"Evaluating {level_name} benchmark...")
        avg_dp = compute_dataset_dp(fds, "train")
        print(f"Results for {level_name}: Avg Dataset DP={avg_dp:.4f}")

        results_summary.append({
            "level": level_name,
            "avg_dp": avg_dp
        })
        
        # Save simple evaluation CSV
        eval_df = pd.DataFrame([{"level": level_name, "avg_dp": avg_dp}])
        eval_path = f"{output_base}/{level_name}_evaluation.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"Evaluation saved to {eval_path}\n")

    print("Summary:")
    for res in results_summary:
        print(res)

if __name__ == "__main__":
    create_benchmarks()
