import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.metrics.fairness import compute_multi_fairness
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def create_value_plot(
    df: pd.DataFrame,
    y_label: str,
    title: str,
    attribute: str,
    font_size_labels: int = 22,
    font_size_title: int = 22,
    font_size_ticks: int = 22,
    file_name: str | None = None,
    save: bool = False,
    save_path: str | None = None,
    custom_labels: dict | None = None,
    jitter_amount: float = 0.1,
):
    fig, _ax = plt.subplots(figsize=(6, 6))

    # Define a color-blind-friendly color
    marker_face_color = "#56B4E9"  # Blue from ColorBrewer Set2
    marker_edge_color = "black"  # Black edges for visibility

    unique_vals = sorted(df["Value"].unique())
    rng = np.random.default_rng()
    for val in unique_vals:
        subset = df[df["Value"] == val]
        y_vals = subset[attribute]
        x_base = int(float(val))
        jitter = rng.uniform(-jitter_amount, jitter_amount, len(y_vals))
        x_vals = [x_base + j for j in jitter]
        plt.scatter(
            x_vals,
            y_vals,
            facecolors=marker_face_color,
            edgecolors=marker_edge_color,
            marker="o",
            s=200,
            linewidths=1.2,
            alpha=0.7,
        )

    plt.xlabel("Sensitive Group Value", fontsize=font_size_labels)
    plt.ylabel(y_label, fontsize=font_size_labels)
    plt.title(title, fontsize=font_size_title)

    if custom_labels:
        ticks = sorted(custom_labels.keys())
        labels = [custom_labels[t] for t in ticks]
        plt.xticks(ticks=ticks, labels=labels, fontsize=font_size_ticks)
    else:
        plt.xticks(ticks=unique_vals, labels=[str(int(float(v))) for v in unique_vals], fontsize=font_size_ticks)

    plt.yticks(fontsize=font_size_ticks)
    plt.grid(visible=True)

    if save:
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        elif file_name:
            plt.savefig(f"{file_name}.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

def run_evaluation_and_plot():
    num_clients = 50
    output_dir = Path("./cross_silo_value/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "drop_mean": 0.5,
        "drop_std": 0.05,
        "flip_mean_0": 0.4,
        "flip_mean_1": 0.4,
        "flip_std": 0.02,
        "target": 0.30,
    }

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
    )
    
    fds.prepare()
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear")
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42)
    else:
        print("Warning: XGBoost is not available in this environment. Only LogisticRegression will be plotted.")

    for model_name, model in models.items():
        print(f"Evaluating Model Fairness for Marital_status using {model_name}...")
        results = compute_multi_fairness(
            partitioner=fds.partitioners["train_train"],
            partitioner_test=fds.partitioners["train_test"],
            model=model,
            sens_atts=["Marital_status"],
            fairness_metric="DP",
            label_name="occupation_binary",
            fds=fds,
            split="train_train",
            test_split="train_test",
            size_unit="attribute"
        )
        
        # Process results for create_value_plot
        plot_df = pd.DataFrame()
        plot_df["DP_MAR"] = results["Marital_status_DP"]
        plot_df["Value"] = results["Marital_status_val"].apply(lambda x: int(float(x.split("_")[-2])))
        
        custom_xticklabels = {1: "Never", 2: "Married", 3: "Divorced", 4: "Widowed"}
        
        print(f"Generating Value Bias Distribution Plot for {model_name}...")
        safe_model_name = model_name.lower().replace(" ", "_")
        create_value_plot(
            plot_df,
            y_label="Dem. Disparity",
            title="Value Bias Distribution (MAR)",
            attribute="DP_MAR",
            save=True,
            save_path=str(output_dir / f"value_dist_marital_status_{safe_model_name}.pdf"),
            custom_labels=custom_xticklabels,
        )
        print(f"Plot saved to {output_dir / f'value_dist_marital_status_{safe_model_name}.pdf'}")

if __name__ == "__main__":
    run_evaluation_and_plot()
