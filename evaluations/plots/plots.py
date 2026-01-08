import argparse
import json
import os
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

# --- Plotting Functions ---

def compute_differences(df1, df2, join_col="dataset"):
    """
    Computes the difference in fairness metrics between two dataframes.
    df1: Local Model Results
    df2: FL Model Results
    """
    # Ensure join_col is available
    if join_col not in df1.columns:
        df1 = df1.reset_index()
    if join_col not in df2.columns:
        df2 = df2.reset_index()
        
    # Merge to ensure alignment
    merged = pd.merge(df1, df2, on=join_col, suffixes=('_local', '_fl'))
    
    # Compute differences (Local - FL) or (FL - Local)? 
    # Notebook: 'DP_RACE': df1['DP_RACE'] - df2['DP_RACE'] (Local - FL)
    # df1 is usually the centralized/local model in the notebook examples.
    
    diff_df = pd.DataFrame({
        join_col: merged[join_col],
        'DP_RACE': merged['DP_RACE_local'] - merged['DP_RACE_fl'],
        'DP_SEX': merged['DP_SEX_local'] - merged['DP_SEX_fl'],
    })
    
    return diff_df

def bar_plot_differences(df, labels, title,
                        font_size_title=25, font_size_ticks=22, font_size_labels=24, y_axis="",
                        save: bool = False, fig_path: str = "", legend_name: str = "bar_plot_differences_legend"):
    """
    Creates a grouped bar plot of unfairness scores for different sensitive attributes.
    """
    # Ensure we are working with a clean dataframe for melting
    df_plot = df.copy()
    if 'State' not in df_plot.columns and 'dataset' in df_plot.columns:
        df_plot = df_plot.rename(columns={'dataset': 'State'})
    
    # If State is not a column but index
    if 'State' not in df_plot.columns:
        df_plot = df_plot.reset_index().rename(columns={'index': 'State'})

    df_melted = df_plot.melt(id_vars='State',
                              value_vars=['DP_SEX', 'DP_RACE'],
                              var_name='Sensitive Attribute',
                              value_name='Unfairness Score')

    ticks = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(16, 6))

    # Define custom colors
    custom_palette = {"DP_SEX": "#1E88E5", "DP_RACE": "#D81B60"}

    sns.barplot(
        x='State', y='Unfairness Score',
        hue='Sensitive Attribute', data=df_melted,
        palette=custom_palette, ax=ax
    )

    # Title and labels
    ax.set_title(title, fontsize=font_size_title, pad=20)
    ax.set_xlabel('State', fontsize=font_size_labels, labelpad=15)
    ax.set_ylabel(y_axis, fontsize=font_size_labels, labelpad=15)

    # Ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=font_size_ticks)
    ax.tick_params(axis='y', labelsize=font_size_ticks)

    # Get the legend object
    legend = ax.get_legend()
    if legend:
        legend.remove()
        
        # Save separate legend if requested (and if save is True)
        if save:
            fig_legend = plt.figure(figsize=(6, 1))
            ax_legend = fig_legend.add_subplot(111)
            handles, labels_legend = ax.get_legend_handles_labels()
            ax_legend.legend(handles, labels_legend,
                               loc='center',
                               fontsize=font_size_ticks,
                               ncol=2, frameon=False)
            ax_legend.axis('off')
            fig_legend.tight_layout()
            # Construct legend path
            legend_path = os.path.join(os.path.dirname(fig_path), legend_name + ".pdf")
            plt.savefig(legend_path, bbox_inches='tight', dpi=150)
            plt.close(fig_legend)

    if save:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

def local_client_fairness_plot(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    client_column: str = "dataset",
    fairness_column: str = "DP_RACE",
    title: str = "Fairness Before/After Comparison",
    figsize: tuple = (6, 6),
    ylabel: str = "Fairness Value Before",
    xlabel: str = "Fairness Value After",
    title_font_size: int = 25,
    label_font_size: int = 25,
    ticks_font_size: int = 20,
    unfairness_distribution: dict = None,
    legend_labels: dict = None,
    save_path: str = None
) -> plt.Figure:
    """
    Plot a scatter comparison of fairness values from two dataframes,
    coloring points based on state lists (if provided).
    df1: Local/Centralized
    df2: Federated
    """
    
    merged = pd.merge(
        df1[[client_column, fairness_column]].rename(columns={fairness_column: "fairness1"}),
        df2[[client_column, fairness_column]].rename(columns={fairness_column: "fairness2"}),
        on=client_column,
    )
    
    fairness1 = merged["fairness1"]
    fairness2 = merged["fairness2"]
    clients = merged[client_column]
    
    min_val = min(fairness1.min(), fairness2.min())
    max_val = max(fairness1.max(), fairness2.max())
    padding = 0.05 * (max_val - min_val) if max_val != min_val else 0.1

    fig, ax = plt.subplots(figsize=figsize)
    
    race_color = '#D81B60'  # Vivid orange
    sex_color = '#1E88E5'   # Blue
    default_color = '#56B4E9'
    
    race_label = legend_labels.get("race", "Race-Related States") if legend_labels else "Race-Related States"
    sex_label = legend_labels.get("sex", "Sex-Related States") if legend_labels else "Sex-Related States"

    # Lists to store data for each group
    race_x, race_y = [], []
    sex_x, sex_y = [], []
    default_x, default_y = [], []
    
    if unfairness_distribution:
        race_states = unfairness_distribution.get("race_state", [])
        sex_states = unfairness_distribution.get("sex_states", [])
        for i, client in enumerate(clients):
            # Check for partial matches if needed, or exact matches
            # The notebook used exact match: `if client in race_states:`
            # But sometimes datasets have suffixes (e.g. AL_2).
            # Assuming exact match for now as per notebook logic.
            
            # Ensure client is string for comparison
            client_str = str(client)
            
            is_race = client_str in race_states
            is_sex = client_str in sex_states
            
            # Helper to check if base state is in list (e.g. AL_2 -> AL)
            if not is_race and not is_sex:
                 base_client = client_str.split('_')[0]
                 is_race = base_client in race_states
                 is_sex = base_client in sex_states

            if is_race:
                race_x.append(fairness2[i])
                race_y.append(fairness1[i])
            elif is_sex:
                sex_x.append(fairness2[i])
                sex_y.append(fairness1[i])
            else:
                default_x.append(fairness2[i])
                default_y.append(fairness1[i])
    else:
        # No distribution provided, all default
        default_x = list(fairness2)
        default_y = list(fairness1)

    # Plot
    if race_x:
        ax.scatter(race_x, race_y, facecolors=race_color, edgecolors='black', marker='o', s=200, linewidths=1.2, alpha=0.8, label=race_label)
    if sex_x:
        ax.scatter(sex_x, sex_y, facecolors=sex_color, edgecolors='black', marker='o', s=200, linewidths=1.2, alpha=0.8, label=sex_label)
    if default_x:
        ax.scatter(default_x, default_y, facecolors=default_color, edgecolors='black', marker='o', s=200, linewidths=1.2, alpha=0.8)

    # Diagonal line
    ax.plot(
        [min_val - padding, max_val + padding],
        [min_val - padding, max_val + padding],
        linestyle="dotted", color="gray", linewidth=2
    )

    ax.tick_params(axis="both", which="major", labelsize=ticks_font_size)
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_title(title, fontsize=title_font_size)
    ax.grid(True)

    # Save logic
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        # Legend
        if unfairness_distribution:
            fig_legend = plt.figure(figsize=(6, 1))
            ax_legend = fig_legend.add_subplot(111)
            race_patch = mpatches.Patch(facecolor=race_color, edgecolor='black', label=race_label)
            sex_patch = mpatches.Patch(facecolor=sex_color, edgecolor='black', label=sex_label)
            ax_legend.legend(handles=[race_patch, sex_patch], fontsize=24, loc='center', frameon=False, ncol=2)
            ax_legend.axis('off')
            fig_legend.tight_layout()
            legend_path = os.path.join(os.path.dirname(save_path), "legend_blue_red.pdf")
            fig_legend.savefig(legend_path, bbox_inches='tight', dpi=150)
            plt.close(fig_legend)
            
        plt.close(fig)
    else:
        plt.show()

# --- Data Loading ---

def get_fl_experiment(wandb_url, partition_names=None, attribute_name="Test Node", dataset_name=""):
    """
    Fetches FL experiment results from WandB.
    """
    api = wandb.Api()
    run = api.run(wandb_url)
    
    # scan_history returns a generator, convert to list then df or directly df
    history = run.scan_history()
    df = pd.DataFrame(history)
    
    attributes = ["- Third DP NEW.", "- Second DP NEW.", "- First DP NEW.", "- Acc.", "- Group 3"]
    
    # Default Mapping
    mapping = {
        "- Third DP NEW.": "DP_RACE",
        "- Second DP NEW.": "DP_MAR",
        "- First DP NEW.": "DP_SEX",
        "- Acc.": "accuracy",
        "- Group 3": "Value"
    }

    # Custom Mapping for Dutch
    if dataset_name and "dutch" in dataset_name.lower():
        # In Dutch: First=Sex, Second=Marital Status
        # We map Second to DP_RACE (Internal placeholder for Attribute 2)
        mapping = {
            "- Third DP NEW.": "DP_OTHER", 
            "- Second DP NEW.": "DP_RACE", # Marital Status
            "- First DP NEW.": "DP_SEX",   # Sex
            "- Acc.": "accuracy",
            "- Group 3": "Value"
        }
    
    first_node_col = f"{attribute_name} 0 - Third DP NEW."
    if first_node_col not in df.columns:
        # Try old mapping
        attributes = ["- Third Disp.", "- Second Disp.", "- Disp.", "- Acc.", "- Group 3"]
        mapping = {
            "- Third Disp.": "DP_RACE",
            "- Second Disp.": "DP_MAR",
            "- Disp.": "DP_SEX",
            "- Acc.": "accuracy",
            "- Group 3": "Value"
        }
        if dataset_name and "dutch" in dataset_name.lower():
             mapping = {
                "- Third Disp.": "DP_OTHER",
                "- Second Disp.": "DP_RACE", # Marital
                "- Disp.": "DP_SEX",         # Sex
                "- Acc.": "accuracy",
                "- Group 3": "Value"
            }

    # If partition_names is not provided, detect nodes from columns
    if partition_names is None:
        import re
        node_ids = set()
        # Look for pattern: "Attribute Name <number> - "
        pattern = re.compile(rf"^{re.escape(attribute_name)} (\d+)")
        for col in df.columns:
            match = pattern.match(col)
            if match:
                node_ids.add(match.group(1))
        partition_names = {nid: nid for nid in sorted(node_ids, key=int)}

    results = {}
    
    # Iterate through nodes.
    for node_id_str in partition_names.keys():
        node = int(node_id_str)
        # Check if any attribute column exists for this node
        # If not, skip (maybe this node wasn't in this run?)
        # But we should try to find it.
        
        node_has_data = False
        temp_res = {}
        
        for attribute in attributes:
            current_attribute = f"{attribute_name} {node} {attribute}"
            if current_attribute in df.columns:
                node_has_data = True
                values = df[current_attribute].values
                values = values[~pd.isna(values)]
                if len(values) > 0:
                    temp_res[mapping[attribute]] = values[-1]
                else:
                    temp_res[mapping[attribute]] = np.nan
        
        if node_has_data:
            results[node] = temp_res
            # Handle dataset name: split('_')[0] is common for cross-silo
            # For cross-device it might be the full name. 
            # We'll use the full name from partition_names and let the user handle splitting if needed,
            # or try to be smart.
            # In notebook: `partition_names[str(node)].split("_")[0]` for cross-silo
            # `partition_names[str(node)]` for cross-device.
            # We can use a heuristic: if user passed a flag or if we check the name format.
            # Let's just use the name from partition_names. 
            # IMPORTANT: The matching with Local results depends on this name.
            # Local results (e.g. from JSON) usually have names like "AL", "TX".
            # Cross-device names might be "AL_0", "AL_1".
            
            p_name = partition_names[node_id_str]
            results[node]["dataset"] = p_name

    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Post-processing: Ensure Value is int if it exists
    if "Value" in results_df.columns:
         results_df["Value"] = results_df["Value"].astype(float).astype(int)
         
    return results_df

def load_local_results(file_path):
    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check for Nested Dict format (Dutch style: { "0": {"lr": ...}, "1": ... })
        if isinstance(data, dict) and len(data) > 0:
            first_key = next(iter(data))
            first_val = data[first_key]
            
            # Heuristic: if values are dicts and contain model names or "accuracy" nested
            if isinstance(first_val, dict) and (any(k in first_val for k in ["lr", "xgb", "rf", "svm", "LogisticRegression", "XGBoost"]) or "accuracy" not in first_val):
                 rows = []
                 for dataset_id, models_dict in data.items():
                     if not isinstance(models_dict, dict): continue
                     
                     for model_name, metrics in models_dict.items():
                         if not isinstance(metrics, dict): continue
                         
                         row = {"dataset": dataset_id, "model": model_name}
                         row["accuracy"] = metrics.get("accuracy")
                         
                         # Map Dutch Keys
                         if "sex_binary_fairness" in metrics:
                             row["DP_SEX"] = metrics["sex_binary_fairness"].get("demographic_disparity")
                         
                         if "Marital_status_fairness" in metrics:
                             # Map Marital Status to DP_RACE (as Attribute 2)
                             row["DP_RACE"] = metrics["Marital_status_fairness"].get("demographic_disparity")
                             
                         # Map Standard Keys (if mixed or other datasets)
                         if "DP_RACE" not in row and "race_fairness" in metrics:
                              row["DP_RACE"] = metrics["race_fairness"].get("demographic_disparity")

                         rows.append(row)
                 return pd.DataFrame(rows)
            
            # Standard flat dict of dicts (Notebook style)
            # Check if values are dicts (nested) -> orient='index'
            if isinstance(first_val, dict):
                 df = pd.DataFrame.from_dict(data, orient='index')
                 if "dataset" not in df.columns:
                     df.index.name = "dataset"
                     df = df.reset_index()
                 return df
            else:
                return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
            
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    
    raise ValueError(f"Unsupported file type: {file_path}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate fairness plots for FL experiments.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset (e.g., dutch, adult)")
    parser.add_argument("--experiment_name", required=True, help="Name of the experiment")
    parser.add_argument("--wandb_url", required=True, help="WandB Run URL (e.g., entity/project/runs/run_id)")
    parser.add_argument("--local_results_path", required=True, help="Path to local/centralized results (JSON/CSV)")
    parser.add_argument("--partition_names_path", help="JSON file mapping node IDs to dataset names (optional)")
    parser.add_argument("--states_unfairness_path", help="JSON file containing 'race_state' and 'sex_states' lists")
    parser.add_argument("--output_dir", default="experiments_plots", help="Base directory for saving plots")
    parser.add_argument("--strip_dataset_suffix", action="store_true", help="If set, removes suffixes like '_0' from dataset names in FL results (useful for Cross-Silo)")
    parser.add_argument("--num_clients", type=int, default=50, help="Number of clients for Dutch dataset distribution logic")
    parser.add_argument("--experiment_type", default="baseline", choices=["baseline", "puffle", "reweighing"], help="Type of experiment (baseline, puffle, reweighing)")
    
    args = parser.parse_args()
    
    # 1. Setup Directories
    base_output_dir = os.path.join("experiments_plots", args.dataset_name, args.experiment_name, args.experiment_type)
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"Output directory: {base_output_dir}")

    # 2. Load Partition Names
    partition_names = None
    if args.partition_names_path:
        with open(args.partition_names_path, 'r') as f:
            partition_names = json.load(f)

    # Load States Unfairness if provided
    states_unfairness = None
    legend_labels = None
    
    if args.states_unfairness_path:
        with open(args.states_unfairness_path, 'r') as f:
            states_unfairness = json.load(f)
    elif args.dataset_name and "dutch" in args.dataset_name.lower():
        # Dutch logic: 0 to N/2-1 is Sex, N/2 to N is Marital (mapped to race_state key)
        half_clients = args.num_clients // 2
        states_unfairness = {
            "sex_states": [str(i) for i in range(0, half_clients)],
            "race_state": [str(i) for i in range(half_clients, args.num_clients)]
        }
        legend_labels = {
            "race": "Marital-Biased Client", 
            "sex": "Sex-Biased Client"
        }
    else:
        # Default fallback from notebook (Dutch Cross-Silo Attribute)
        states_unfairness = {
            "sex_states": ['SD', 'IN', 'WV', 'PA', 'IL', 'MI', 'WA', 'TX', 'MO', 'WY', 'TN', 'OK', 'UT', 'ID', 'ND', 'VA', 'AR', 'KS', 'NH', 'OH', 'LA'],
            "race_state": ['AL', 'NM', 'IA', 'MA', 'FL', 'AZ', 'NY', 'AK', 'MS', 'NC', 'GA', 'VT', 'SC', 'NJ', 'CT', 'DE', 'RI', 'WI', 'OR', 'NV', 'NE', 'MN', 'CA', 'MT', 'MD', 'CO', 'HI', 'KY', 'ME']
        }
        legend_labels = {
            "race": "Race-Related States", 
            "sex": "Sex-Related States"
        }

    # 3. Fetch FL Results
    print(f"Fetching FL results from {args.wandb_url}...")
    fl_df = get_fl_experiment(args.wandb_url, partition_names, dataset_name=args.dataset_name)
    
    if args.strip_dataset_suffix:
        fl_df["dataset"] = fl_df["dataset"].apply(lambda x: x.split('_')[0])
    
    print(f"Loaded FL results for {len(fl_df)} nodes.")    
    # 4. Load Local Results
    print(f"Loading local results from {args.local_results_path}...")
    local_df = load_local_results(args.local_results_path)
    print(f"Loaded local results for {len(local_df)} entries.")
    
    # Filter local results to match FL model type? 
    # Notebook filters by model e.g. dp[dp["model"] == "LogisticRegression"]
    # We should probably produce plots for all models found in local results
    
    if "model" in local_df.columns:
        models = local_df["model"].unique()
    else:
        models = ["LocalModel"]
        local_df["model"] = "LocalModel"

    # 5. Generate Plots
    
    # Determine label for FL method based on experiment type
    fl_method_label = "FedAvg"
    if args.experiment_type == "puffle":
        fl_method_label = "Puffle"
    elif args.experiment_type == "reweighing":
        fl_method_label = "Reweighing"

    for model in models:
        model_df = local_df[local_df["model"] == model]
        safe_model_name = model.replace(" ", "_").lower()
        
        # A) Difference Histogram
        # Calculate diffs
        try:
            diff_df = compute_differences(model_df, fl_df)
            # Drop entries where we can't compute diff (e.g. missing match)
            diff_df = diff_df.dropna(subset=["DP_SEX", "DP_RACE"])
            
            if not diff_df.empty:
                plot_path = os.path.join(base_output_dir, f"diff_hist_{safe_model_name}.pdf")
                bar_plot_differences(
                    diff_df, 
                    list(diff_df["dataset"]),
                    title=f"{model} - {fl_method_label} Unfairness Difference", 
                    y_axis="Dem. Disparity Difference",
                    save=True,
                    fig_path=plot_path
                )
                print(f"Saved Difference Histogram: {plot_path}")
            else:
                print(f"No matching data for Difference Histogram for model {model}")
                
        except Exception as e:
            print(f"Error creating difference histogram for {model}: {e}")

        # B) Distribution Dots (Scatter Comparison)
        # Race / Marital
        try:
            plot_path = os.path.join(base_output_dir, f"scatter_race_{safe_model_name}.pdf")
            
            # Dutch specific limits
            x_limit = 0.6 if args.dataset_name and "dutch" in args.dataset_name.lower() else None
            # y_limit = 0.6 if args.dataset_name and "dutch" in args.dataset_name.lower() else None

            local_client_fairness_plot(
                df1=model_df,
                df2=fl_df,
                fairness_column="DP_RACE",
                ylabel=f"{model} Dem. Disparity",
                xlabel=f"{fl_method_label} Dem. Disparity",
                title="MAR Unfairness Distribution" if args.dataset_name and "dutch" in args.dataset_name.lower() else "RACE Unfairness Distribution",
                unfairness_distribution=states_unfairness,
                legend_labels=legend_labels,
                save_path=plot_path,
                # x_lim=x_limit,
                # y_lim=y_limit
            )
            print(f"Saved Scatter Plot (Race/MAR): {plot_path}")
        except Exception as e:
             print(f"Error creating scatter plot (Race) for {model}: {e}")

        # Sex
        try:
            plot_path = os.path.join(base_output_dir, f"scatter_sex_{safe_model_name}.pdf")
            
            # Dutch specific limits
            x_limit = 0.3 if args.dataset_name and "dutch" in args.dataset_name.lower() else None
            # y_limit = 0.3 if args.dataset_name and "dutch" in args.dataset_name.lower() else None

            local_client_fairness_plot(
                df1=model_df,
                df2=fl_df,
                fairness_column="DP_SEX",
                ylabel=f"{model} Dem. Disparity",
                xlabel=f"{fl_method_label} Dem. Disparity",
                title="SEX Unfairness Distribution",
                unfairness_distribution=states_unfairness,
                legend_labels=legend_labels,
                save_path=plot_path,
                # x_lim=x_limit,
                # y_lim=y_limit
            )
            print(f"Saved Scatter Plot (Sex): {plot_path}")
        except Exception as e:
             print(f"Error creating scatter plot (Sex) for {model}: {e}")

if __name__ == "__main__":
    main()