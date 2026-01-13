import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# --- Style Constants (Matching plots.py) ---
FONT_SIZE_TITLE = 25
FONT_SIZE_LABELS = 24
FONT_SIZE_TICKS = 22
LEGEND_FONT_SIZE = 22

# Colors
COLOR_MALE = "#1E88E5"      # Blue (matches DP_SEX)
COLOR_HAIR = "#D81B60"      # Pink (matches DP_RACE)
COLOR_DEFAULT = "#56B4E9"   # Light Blue

# Value Colors
VAL_COLORS = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'gray'}
VAL_LABELS = {0: 'Black Hair', 1: 'Blond Hair', 2: 'Brown Hair', 3: 'Gray Hair'}

def setup_style():
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': FONT_SIZE_LABELS,
        'axes.titlesize': FONT_SIZE_TITLE,
        'xtick.labelsize': FONT_SIZE_TICKS,
        'ytick.labelsize': FONT_SIZE_TICKS,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

def save_legend(handles, labels, output_dir, filename, ncol=2):
    """Saves a legend to a separate file."""
    fig_legend = plt.figure(figsize=(6, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles, labels, loc='center', fontsize=LEGEND_FONT_SIZE, ncol=ncol, frameon=False)
    ax_legend.axis('off')
    fig_legend.tight_layout()
    legend_path = os.path.join(output_dir, filename)
    fig_legend.savefig(legend_path, bbox_inches='tight', dpi=150)
    plt.close(fig_legend)
    print(f"Saved legend to {legend_path}")

def plot_attribute_unfairness(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Plots the unfairness of each client for Male and Hair Color attributes.
    Saves the plot and a separate legend.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Check columns
    required_cols = ["Male_DP", "hair_color_DP"]
    if not all(col in df.columns for col in required_cols):
        print(f"Missing columns in {csv_path}. Available: {df.columns}")
        return

    # Rename for display
    df_plot = df.rename(columns={
        "Male_DP": "Male",
        "hair_color_DP": "Hair Color",
        "Partition ID": "Client ID"
    })
    
    # Melt
    df_melted = df_plot.melt(
        id_vars=["Client ID"],
        value_vars=["Male", "Hair Color"],
        var_name="Attribute",
        value_name="Unfairness Score" # Match plots.py y-axis label
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    custom_palette = {"Male": COLOR_MALE, "Hair Color": COLOR_HAIR}
    
    sns.barplot(
        x="Client ID",
        y="Unfairness Score",
        hue="Attribute",
        data=df_melted,
        palette=custom_palette,
        ax=ax
    )
    
    ax.set_title(f"Attribute Unfairness per Client {title_suffix}", fontsize=FONT_SIZE_TITLE, pad=20)
    ax.set_xlabel("Client ID", fontsize=FONT_SIZE_LABELS, labelpad=15)
    ax.set_ylabel("Dem. Disparity", fontsize=FONT_SIZE_LABELS, labelpad=15)
    
    # Ticks
    unique_clients = df_plot["Client ID"].unique()
    n = len(unique_clients)
    if n > 50:
        ticks = range(0, n, 10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(c) for c in unique_clients[list(ticks)]], rotation=0, fontsize=FONT_SIZE_TICKS)
    else:
        ax.set_xticks(range(n))
        ax.set_xticklabels(unique_clients, rotation=90, fontsize=FONT_SIZE_TICKS)
        
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove legend from plot
    legend = ax.get_legend()
    if legend:
        legend.remove()
    
    # Save Plot
    output_path = os.path.join(output_dir, f"{file_prefix}_bar.pdf")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved attribute bar plot to {output_path}")
    
    # Save Legend Separately
    handles = [
        mpatches.Patch(color=COLOR_MALE, label='Male'),
        mpatches.Patch(color=COLOR_HAIR, label='Hair Color')
    ]
    save_legend(handles, ['Male', 'Hair Color'], output_dir, f"{file_prefix}_legend.pdf")

def plot_attribute_scatter(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Scatter plot: Male Unfairness (X) vs Hair Color Unfairness (Y).
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    x_col = "Male_DP"
    y_col = "hair_color_DP"
    
    if x_col not in df.columns or y_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 8)) # Square for scatter often looks better
    
    ax.scatter(
        df[x_col],
        df[y_col],
        facecolors=COLOR_DEFAULT,
        edgecolors='black',
        marker='o',
        s=200,
        linewidths=1.2,
        alpha=0.8
    )
    
    # Diagonal line
    min_val = min(df[x_col].min(), df[y_col].min())
    max_val = max(df[x_col].max(), df[y_col].max())
    padding = 0.05 * (max_val - min_val) if max_val != min_val else 0.1
    
    # Limits
    lim_min = 0 # Force positive values only
    lim_max = max_val + padding
    
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="dotted", color="gray", linewidth=2)
    
    ax.set_xlabel("Male Unfairness (DP)", fontsize=FONT_SIZE_LABELS, labelpad=10)
    ax.set_ylabel("Hair Color Unfairness (DP)", fontsize=FONT_SIZE_LABELS, labelpad=10)
    ax.set_title(f"Attribute Bias Distribution {title_suffix}", fontsize=FONT_SIZE_TITLE, pad=20)
    
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS)
    ax.grid(True)
    
    output_path = os.path.join(output_dir, f"{file_prefix}_scatter.pdf")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved attribute scatter plot to {output_path}")

def plot_value_lollipop_sorted(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Lollipop Chart: Clients sorted by max unfairness.
    Line from 0 to value, dot at top colored by group.
    Cleaner than bar/scatter.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Identify hair color columns
    hair_cols = [c for c in df.columns if "hair_color" in c and c not in ["hair_color_DP", "hair_color_val"]]
    
    if not hair_cols: return

    # Calculate Max DP
    if "Max_DP" in df.columns:
        max_dp = df["Max_DP"]
        idxmax = df[hair_cols].idxmax(axis=1)
    else:
        max_dp = df[hair_cols].max(axis=1)
        idxmax = df[hair_cols].idxmax(axis=1)
    
    def get_value_from_col(col_name):
        if not isinstance(col_name, str): return -1
        if "_0_" in col_name or col_name.endswith("_0"): return 0
        if "_1_" in col_name or col_name.endswith("_1"): return 1
        if "_2_" in col_name or col_name.endswith("_2"): return 2
        if "_3_" in col_name or col_name.endswith("_3"): return 3
        return 4 

    values = idxmax.apply(get_value_from_col)
    
    plot_df = pd.DataFrame({
        "Client ID": df["Partition ID"],
        "Demographic Parity": max_dp,
        "Value": values
    })
    
    # Sort
    plot_df = plot_df.sort_values("Demographic Parity", ascending=False).reset_index(drop=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Draw Lines (stem)
    # vlines(x, ymin, ymax)
    # We color the stems gray or black for simplicity, or match the head color.
    # Matching head color looks better.
    
    colors = [VAL_COLORS.get(v, 'gray') for v in plot_df["Value"]]
    
    ax.vlines(x=plot_df.index, ymin=0, ymax=plot_df["Demographic Parity"], color=colors, alpha=0.4, linewidth=1)
    
    # Draw Dots (head)
    ax.scatter(
        plot_df.index, 
        plot_df["Demographic Parity"], 
        c=colors, 
        s=40, # Smaller than previous scatter, cleaner
        zorder=2
    )
    
    ax.set_title(f"Value Unfairness Profile (Lollipop) {title_suffix}", fontsize=FONT_SIZE_TITLE, pad=20)
    ax.set_xlabel("Clients (Sorted by Unfairness)", fontsize=FONT_SIZE_LABELS, labelpad=15)
    ax.set_ylabel("Max Dem. Disparity", fontsize=FONT_SIZE_LABELS, labelpad=15)
    
    ax.set_xlim(-0.5, len(plot_df)-0.5)
    ax.set_ylim(bottom=0)
    
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax.tick_params(axis='x', labelsize=FONT_SIZE_TICKS)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, f"{file_prefix}_lollipop.pdf")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved lollipop plot to {output_path}")

def plot_value_boxplot_grouped(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Box Plot grouped by Dominant Hair Color Value.
    Shows the distribution of max unfairness *for clients dominated by that value*.
    """
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    hair_cols = [c for c in df.columns if "hair_color" in c and c not in ["hair_color_DP", "hair_color_val"]]
    if not hair_cols: return

    if "Max_DP" in df.columns:
        max_dp = df["Max_DP"]
        idxmax = df[hair_cols].idxmax(axis=1)
    else:
        max_dp = df[hair_cols].max(axis=1)
        idxmax = df[hair_cols].idxmax(axis=1)
    
    def get_value_from_col(col_name):
        if not isinstance(col_name, str): return -1
        if "_0_" in col_name or col_name.endswith("_0"): return 0
        if "_1_" in col_name or col_name.endswith("_1"): return 1
        if "_2_" in col_name or col_name.endswith("_2"): return 2
        if "_3_" in col_name or col_name.endswith("_3"): return 3
        return 4

    values = idxmax.apply(get_value_from_col)
    
    plot_df = pd.DataFrame({
        "Demographic Parity": max_dp,
        "Value": values
    })
    
    # Filter only valid values 0-3 and ensure int type for palette matching
    plot_df = plot_df[plot_df["Value"].isin([0,1,2,3])].copy()
    plot_df["Value"] = plot_df["Value"].astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Box Plot
    sns.boxplot(
        x="Value", 
        y="Demographic Parity", 
        data=plot_df, 
        ax=ax,
        palette=VAL_COLORS,
        width=0.5,
        boxprops=dict(alpha=0.6),
        showfliers=False,
        hue="Value", # Fix FutureWarning
        legend=False
    )
    
    # Strip Plot on top for individual points
    sns.stripplot(
        x="Value", 
        y="Demographic Parity", 
        data=plot_df, 
        ax=ax,
        palette=VAL_COLORS,
        hue="Value", # Fix FutureWarning
        legend=False,
        jitter=True,
        size=6,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8
    )
    
    ax.set_title(f"Max Unfairness Distribution by Dominant Group {title_suffix}", fontsize=FONT_SIZE_TITLE, pad=20)
    ax.set_xlabel("Dominant Hair Color", fontsize=FONT_SIZE_LABELS, labelpad=15)
    ax.set_ylabel("Max Dem. Disparity", fontsize=FONT_SIZE_LABELS, labelpad=15)
    
    ax.set_xticklabels([VAL_LABELS[0], VAL_LABELS[1], VAL_LABELS[2], VAL_LABELS[3]], fontsize=FONT_SIZE_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax.set_ylim(bottom=0)
    
    output_path = os.path.join(output_dir, f"{file_prefix}_boxplot.pdf")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved boxplot to {output_path}")

def plot_value_distribution_scatter(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Jittered Scatter Plot: Value (X) vs DP (Y).
    Shows distribution of unfairness per value group (ALL unfairness values, not just max).
    """
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    data_points = []
    
    for col in df.columns:
        if "hair_color" not in col: continue
        if col in ["hair_color_DP", "hair_color_val", "Max_DP"]: continue
        val = -1
        if "_0_" in col or col.endswith("_0"): val = 0
        elif "_1_" in col or col.endswith("_1"): val = 1
        elif "_2_" in col or col.endswith("_2"): val = 2
        elif "_3_" in col or col.endswith("_3"): val = 3
        
        if val != -1:
            for dp_val in df[col]:
                data_points.append({"Value": val, "DP": dp_val})
                
    if not data_points: return
    plot_df = pd.DataFrame(data_points)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    jitter_amount = 0.15
    
    for val in sorted(plot_df["Value"].unique()):
        subset = plot_df[plot_df["Value"] == val]
        y_vals = subset["DP"]
        x_base = int(val)
        x_jittered = np.random.uniform(x_base - jitter_amount, x_base + jitter_amount, len(y_vals))
        
        ax.scatter(x_jittered, y_vals, facecolors=VAL_COLORS[val], edgecolors='black', marker='o', s=150, linewidths=1.0, alpha=0.6)
        
    ax.set_title(f"Value Bias Distribution (All Groups) {title_suffix}", fontsize=FONT_SIZE_TITLE, pad=20)
    ax.set_xlabel("Hair Color Group", fontsize=FONT_SIZE_LABELS, labelpad=15)
    ax.set_ylabel("Dem. Disparity", fontsize=FONT_SIZE_LABELS, labelpad=15)
    
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([VAL_LABELS[0], VAL_LABELS[1], VAL_LABELS[2], VAL_LABELS[3]], fontsize=FONT_SIZE_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    
    output_path = os.path.join(output_dir, f"{file_prefix}_scatter_dist.pdf")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved value scatter distribution to {output_path}")

def main():
    setup_style()
    
    base_dir = "../../datasets/celeba"
    output_root = "evaluations/plots/experiments_plots/celeba"
    
    # 1. Attribute Plots
    attr_out_dir = os.path.join(output_root, "attribute")
    os.makedirs(attr_out_dir, exist_ok=True)
    
    # Cross-Device Attribute
    csv_path = os.path.join(base_dir, "cross_device_attribute", "medium_evaluation.csv")
    plot_attribute_unfairness(csv_path, attr_out_dir, "cross_device", "(Cross-Device)")
    plot_attribute_scatter(csv_path, attr_out_dir, "cross_device", "(Cross-Device)")
    
    # Cross-Silo Attribute
    csv_path = os.path.join(base_dir, "cross_silo_attribute", "medium_evaluation.csv")
    plot_attribute_unfairness(csv_path, attr_out_dir, "cross_silo", "(Cross-Silo)")
    plot_attribute_scatter(csv_path, attr_out_dir, "cross_silo", "(Cross-Silo)")

    # 2. Value Plots
    val_out_dir = os.path.join(output_root, "value")
    os.makedirs(val_out_dir, exist_ok=True)
    
    # Cross-Device Value
    csv_path = os.path.join(base_dir, "cross_device_value", "medium_evaluation.csv")
    plot_value_lollipop_sorted(csv_path, val_out_dir, "cross_device", "(Cross-Device)")
    plot_value_boxplot_grouped(csv_path, val_out_dir, "cross_device", "(Cross-Device)")
    # Keep the previous distribution scatter for completeness
    plot_value_distribution_scatter(csv_path, val_out_dir, "cross_device", "(Cross-Device)")
    
    # Cross-Silo Value
    csv_path = os.path.join(base_dir, "cross_silo_value", "medium_evaluation.csv")
    plot_value_lollipop_sorted(csv_path, val_out_dir, "cross_silo", "(Cross-Silo)")
    plot_value_boxplot_grouped(csv_path, val_out_dir, "cross_silo", "(Cross-Silo)")
    plot_value_distribution_scatter(csv_path, val_out_dir, "cross_silo", "(Cross-Silo)")

if __name__ == "__main__":
    main()
