"""
Creation script for Cross-Silo Attribute Imbalanced Benchmarking Datasets for CelebA.
Dataset: flwrlabs/celeba
Scenario: Cross-Silo (50 clients)
Target DP Level: Medium (0.30)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import json
import base64
from datasets import load_dataset, concatenate_datasets
from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias
from FeDa4Fair.visualization.plots import plot_multi_attribute_fairness

def add_hair_color(df):
    """
    Adds 'hair_color' column:
    0 (Dark): Black_Hair, Brown_Hair
    1 (Light): Blond_Hair, Gray_Hair, and others (etc.)
    """
    # Vectorized operation
    is_dark = df['Black_Hair'] | df['Brown_Hair']
    df['hair_color'] = np.where(is_dark, 0, 1)
    return df

def create_benchmarks():
    num_clients = 50
    output_base = "datasets/celeba/cross_silo_attribute"
    img_dict_path = "datasets/celeba/celeba_img_dict.json"
    
    # 1. Load and Preprocess Data
    print("Loading CelebA dataset...")
    ds_dict = load_dataset("flwrlabs/celeba")
    
    ds_merged = concatenate_datasets(list(ds_dict.values()))
    
    print("Adding image IDs...")
    ds_merged = ds_merged.add_column("image_id", range(len(ds_merged)))

    # Check if image dict exists
    if not os.path.exists(img_dict_path):
        print(f"Creating image dictionary at {img_dict_path}...")
        img_map = {}
        for item in ds_merged:
            idx = item['image_id']
            img = item['image']
            b = io.BytesIO()
            img.save(b, format="PNG")
            b64_str = base64.b64encode(b.getvalue()).decode('utf-8')
            img_map[idx] = b64_str
        
        print("Saving JSON...")
        os.makedirs(os.path.dirname(img_dict_path), exist_ok=True)
        with open(img_dict_path, "w") as f:
            json.dump(img_map, f)
        print("JSON saved.")
        del img_map
    else:
        print(f"Image dictionary found at {img_dict_path}, skipping creation.")

    print("Dropping image column...")
    ds_merged = ds_merged.remove_columns("image")
    
    print("Converting to Pandas for preprocessing...")
    df = ds_merged.to_pandas()
    
    print("Adding 'hair_color' attribute...")
    df = add_hair_color(df)

    # Tuning params for Target DP ~0.30
    level_name = "medium"
    print(f"Creating {level_name} benchmark (Target DP ~0.30)...")

    half_clients = num_clients // 2
    
    # Tuned parameters:
    # Male: 0.28 flip -> ~0.30 DP
    # Hair: 0.60 flip -> ~0.30 DP (Hair needed stronger bias injection)
    
    group_configs = [
        {
            "group_id": "unfair_male_fair_hair",
            "num_clients": half_clients,
            "configs": [
                {
                    "attribute": "Male",
                    "value": 1, 
                    "drop_mean": 0.3, "drop_std": 0.05,
                    "flip_mean": 0.28, "flip_std": 0.02,
                    "mitigate": False # Inject bias
                },
                {
                    "attribute": "hair_color",
                    "value": 0, 
                    "mitigate": True # Fair for hair
                }
            ]
        },
        {
            "group_id": "fair_male_unfair_hair",
            "num_clients": num_clients - half_clients,
            "configs": [
                {
                    "attribute": "Male",
                    "value": 1, 
                    "mitigate": True # Fair for Male
                },
                {
                    "attribute": "hair_color",
                    "value": 0, 
                    "drop_mean": 0.3, "drop_std": 0.05,
                    "flip_mean": 0.60, "flip_std": 0.02, # Increased significantly
                    "mitigate": False # Inject bias
                }
            ]
        }
    ]

    mod_dict = generate_multiobjective_bias(num_clients, group_configs)

    fds = FairFederatedDataset(
        dataset="flwrlabs/celeba", 
        preloaded_data=df, 
        split="all",
        partitioners={"train": num_clients},
        label_name="Smiling",
        sensitive_attributes=["Male", "hair_color"],
        modification_dict=mod_dict,
        fl_setting="cross-silo",
        perc_train_val_test=[0.8, 0.2],
        path=f"{output_base}/{level_name}"
    )

    fds.prepare()

    # Evaluation
    print(f"Evaluating {level_name} benchmark...")
    
    sens_atts = ["Male", "hair_color"]
    train_key = "train_train"
    if train_key not in fds.partitioners:
        train_key = "train"

    # Evaluate Data DP
    fig, ax, results_dp = plot_multi_attribute_fairness(
        partitioner=fds.partitioners[train_key],
        partitioner_test=fds.partitioners[train_key],
        model=None,
        sens_atts=sens_atts,
        fairness_metric="DP",
        label_name="Smiling",
        fds=fds,
        split="train_train",
        size_unit="attribute" 
    )
    
    fig.savefig(f"{output_base}/{level_name}_DP.png")
    results_dp.to_csv(f"{output_base}/{level_name}_evaluation.csv")
    plt.close(fig)
    
    print(f"Evaluation saved to {output_base}/{level_name}_evaluation.csv\n")
    
    if "DP" in results_dp.columns:
        print("Metric Columns:", results_dp.columns)
        numeric_cols = results_dp.select_dtypes(include=np.number).columns
        print(f"Average DP across clients: {results_dp[numeric_cols].mean().mean():.4f}")

if __name__ == "__main__":
    create_benchmarks()
