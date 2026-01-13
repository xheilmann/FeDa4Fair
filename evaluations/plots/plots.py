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

def create_value_plot(
    df: pd.DataFrame,
    y_label: str,
    title: str,
    attribute: str,
    font_size_labels: int = 22,
    font_size_title: int = 22,
    font_size_ticks: int = 22,
    file_name: str = None,
    save: bool = False,
    save_path: str = None,
    custom_labels: dict = None,
    jitter_amount: float = 0.1
):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Define a color-blind-friendly color
    marker_face_color = '#56B4E9'  # Blue from ColorBrewer Set2
    marker_edge_color = 'black'    # Black edges for visibility

    unique_vals = sorted(df['Value'].unique())
    for val in unique_vals:
        subset = df[df['Value'] == val]
        y_vals = subset[attribute]
        x_base = int(float(val))
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(y_vals))
        x_vals = [x_base + j for j in jitter]
        plt.scatter(
            x_vals,
            y_vals,
            facecolors=marker_face_color,
            edgecolors=marker_edge_color,
            marker='o',
            s=200,
            linewidths=1.2,
            alpha=0.7
        )

    plt.xlabel('Sensitive Group Value', fontsize=font_size_labels)
    plt.ylabel(y_label, fontsize=font_size_labels)
    plt.title(title, fontsize=font_size_title)

    if custom_labels:
        ticks = sorted(custom_labels.keys())
        labels = [custom_labels[t] for t in ticks]
        plt.xticks(ticks=ticks, labels=labels, fontsize=font_size_ticks)
    else:
        plt.xticks(ticks=unique_vals, labels=[str(int(float(v))) for v in unique_vals], fontsize=font_size_ticks)
    
    plt.yticks(fontsize=font_size_ticks)
    plt.grid(True)

    if save:
        # Use save_path if provided, otherwise file_name in ./images/
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        elif file_name:
            plt.savefig(f"./images/{file_name}.pdf", bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

def visualize_value_change(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    sensitive_col: str = "DP_RACE",
    value_col: str = "Value",
    font_size: int = 26,
    ticks_font_size: int = 20,
    title: str = "",
    y_label: str = "",
    initial_state: str = "",
    jitter_amount: float = 0.1,
    legend_filename: str = "value_change_legend.pdf",
    save_path: str = None,
    custom_labels: dict = None
) -> plt.Figure:
    """
    Visualizes the change in a specified sensitive column across different value
    categories between two dataframes with jittered dots and a separate legend file.
    Arrows are drawn only if the 'Value' is different between the two connected states.
    """
    # Merge the two dataframes based on the common columns
    merged_df = pd.merge(df1, df2, on="dataset", suffixes=('_df1', '_df2'))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Styling for the initial state scatter plot
    marker_face_color = '#004D40'
    marker_edge_color = 'black'
    marker_size = 200
    marker_linewidth = 1.2

    # Arrow styling
    arrow_head_width = 0.05
    arrow_head_length = 0.05
    arrow_alpha = 0.7
    arrow_linewidth = 1.5
    arrow_color = 'gray'

    # Store jittered positions: {dataset_id: (x, y)}
    pos_map_df1 = {}
    pos_map_df2 = {}

    # Plot the initial state with jitter
    unique_vals = sorted(df1[value_col].unique())
    for val in unique_vals:
        subset = df1[df1[value_col] == val]
        y_vals = subset[sensitive_col].astype(float)
        x_base = int(float(val))
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(y_vals))
        x_vals = [x_base + j for j in jitter]
        
        # Store positions
        for i, dataset_id in enumerate(subset['dataset']):
            pos_map_df1[dataset_id] = (x_vals[i], y_vals.values[i])
            
        plt.scatter(
            x_vals,
            y_vals,
            facecolors=marker_face_color,
            edgecolors=marker_edge_color,
            marker='o',
            s=marker_size,
            linewidths=marker_linewidth,
            label=initial_state if val == df1[value_col].unique()[0] else "",  # Label only once
            zorder=2,  # Ensure initial state points are on top of arrows
            alpha=0.6
        )

    # Plot the final state points with jitter
    for val in sorted(df2[value_col].unique()):
        subset = df2[df2[value_col] == val]
        y_vals = subset[sensitive_col].astype(float)
        x_base = int(float(val))
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(y_vals))
        x_vals = [x_base + j for j in jitter]
        
        # Store positions
        for i, dataset_id in enumerate(subset['dataset']):
            pos_map_df2[dataset_id] = (x_vals[i], y_vals.values[i])

        plt.scatter(
            x_vals,
            y_vals,
            s=200,
            color='#FFC107',
            edgecolor='black',
            label='FedAVG' if val == df2[value_col].unique()[0] else "",  # Label only once
            zorder=2,  # Ensure final state points are on top of arrows
            alpha=0.8
        )

    # Draw arrows based on the 'dataset' identifier
    for index, row in merged_df.iterrows():
        dataset_id = row['dataset']
        
        if dataset_id in pos_map_df1 and dataset_id in pos_map_df2:
            initial_x, initial_y = pos_map_df1[dataset_id]
            final_x, final_y = pos_map_df2[dataset_id]

            # Only draw arrow if there is a significant change in position
            if abs(initial_x - final_x) > 0.01 or abs(initial_y - final_y) > 0.001:
                plt.arrow(initial_x, initial_y, final_x - initial_x, final_y - initial_y,
                          head_width=arrow_head_width,
                          head_length=arrow_head_length,
                          fc=arrow_color,
                          ec=arrow_color,
                          alpha=arrow_alpha,
                          linewidth=arrow_linewidth,
                          length_includes_head=True,
                          zorder=1)

    # Customize the x-axis ticks
    if custom_labels:
        ticks = sorted(custom_labels.keys())
        labels = [custom_labels[t] for t in ticks]
        plt.xticks(ticks=ticks, labels=labels, fontsize=ticks_font_size)
    else:
        plt.xticks(ticks=unique_vals, labels=[str(int(float(v))) for v in unique_vals], fontsize=ticks_font_size)

    plt.yticks(fontsize=ticks_font_size)
    plt.xlabel('Sensitive Group Value', fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.grid(True)

    # Create a separate figure for the legend
    if save_path:
        fig_legend = plt.figure(figsize=(6, 1))
        ax_legend = fig_legend.add_subplot(111)
        initial_patch = mpatches.Patch(facecolor='#004D40', edgecolor='black', alpha=0.6, label=initial_state)
        fedavg_patch = mpatches.Patch(facecolor='#FFC107', edgecolor='black', alpha=0.8, label=str(title).split(" ")[-1] if "FedAvg" not in title else "FedAvg") # Rough fallback
        # Better label handling needed, but for now:
        fedavg_patch = mpatches.Patch(facecolor='#FFC107', edgecolor='black', alpha=0.8, label='FedAVG')

        ax_legend.legend(handles=[initial_patch, fedavg_patch], fontsize=20, loc='center', frameon=False, ncol=2)
        ax_legend.axis('off')
        
        legend_path = os.path.join(os.path.dirname(save_path), legend_filename)
        fig_legend.tight_layout()
        fig_legend.savefig(legend_path)
        plt.close(fig_legend)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return fig
    else:
        plt.tight_layout()
        plt.show()
        return fig

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
    save_path: str = None,
    x_lim: float = None,
    y_lim: float = None
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
    diag_max = max(x_lim, y_lim) if (x_lim and y_lim) else (max_val + padding)
    diag_min = 0 # Usually fairness metrics start at 0
    
    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        linestyle="dotted", color="gray", linewidth=2
    )

    ax.tick_params(axis="both", which="major", labelsize=ticks_font_size)
    
    if x_lim:
        ax.set_xlim(0, x_lim)
    else:
        ax.set_xlim(min_val - padding, max_val + padding)
        
    if y_lim:
        ax.set_ylim(0, y_lim)
    else:
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

def scatter_fairness_plot(
    df1: pd.DataFrame,
    client_column: str = "Partition ID",
    fairness_column_Y: str = "RAC1P_DP",
    fairness_column_X: str = "RAC1P_DP",
    title: str = "Fairness Comparison",
    figsize: tuple = (6, 6),
    ylabel: str = "Fairness Metric Y",
    xlabel: str = "Fairness Metric X",
    title_font_size: int = 25,
    label_font_size: int = 25,
    ticks_font_size: int = 20,
    unfairness_distribution: dict = None,
    legend_labels: dict = None,
    legend_filename: str = "fairness_plot_legend.png",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot a scatter comparison of two fairness metrics from the same DataFrame,
    coloring points based on state lists, with the legend saved separately.
    """
    # assert df1[client_column].is_unique, "The client ID column must be unique."

    fairness_x = df1[fairness_column_X]
    fairness_y = df1[fairness_column_Y]
    clients = df1[client_column]

    min_val = min(fairness_x.min(), fairness_y.min())
    max_val = max(fairness_x.max(), fairness_y.max())
    padding = 0.05 * (max_val - min_val) if max_val != min_val else 0.1

    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for the two groups
    race_color = '#D81B60'  # Vivid orange
    sex_color = '#1E88E5'   # Blue
    
    race_label = legend_labels.get("race", "Race-Related States") if legend_labels else "Race-Related States"
    sex_label = legend_labels.get("sex", "Sex-Related States") if legend_labels else "Sex-Related States"

    # Lists to store data for each group
    race_x = []
    race_y = []
    sex_x = []
    sex_y = []

    if unfairness_distribution:
        race_states = unfairness_distribution.get("race_state", [])
        sex_states = unfairness_distribution.get("sex_states", [])

        for i, client in enumerate(clients):
            client_str = str(client)
            is_race = client_str in race_states
            is_sex = client_str in sex_states
            
            if not is_race and not is_sex:
                 base_client = client_str.split('_')[0]
                 is_race = base_client in race_states
                 is_sex = base_client in sex_states
            
            if is_race:
                race_x.append(fairness_x.iloc[i])
                race_y.append(fairness_y.iloc[i])
            elif is_sex:
                sex_x.append(fairness_x.iloc[i])
                sex_y.append(fairness_y.iloc[i])
            else:
                # We are not including the 'other' states
                pass
    else:
        # If no unfairness_distribution is provided, plot all points with a default color
        ax.scatter(
            fairness_x,
            fairness_y,
            facecolors='#56B4E9',
            edgecolors='black',
            marker='o',
            s=200,
            linewidths=1.2,
            alpha=0.8,
        )
        ax.plot(
            [min_val - padding, max_val + padding],
            [min_val - padding, max_val + padding],
            linestyle="dotted",
            color="gray",
            linewidth=2
        )
        ax.tick_params(axis="both", which="major", labelsize=ticks_font_size)
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        ax.set_title(title, fontsize=title_font_size)
        ax.grid(True)
        
        if save_path:
             plt.tight_layout()
             plt.savefig(save_path, bbox_inches='tight', dpi=150)
             plt.close(fig)
        else:
             plt.tight_layout()
             plt.show()
        return fig

    # Scatter plot for race-related states
    ax.scatter(
        race_x,
        race_y,
        facecolors=race_color,
        edgecolors='black',
        marker='o',
        s=200,
        linewidths=1.2,
        alpha=0.8,
        label=race_label
    )

    # Scatter plot for sex-related states
    ax.scatter(
        sex_x,
        sex_y,
        facecolors=sex_color,
        edgecolors='black',
        marker='o',
        s=200,
        linewidths=1.2,
        alpha=0.8,
        label=sex_label
    )

    ax.plot(
        [min_val - padding, max_val + padding],
        [min_val - padding, max_val + padding],
        linestyle="dotted",
        color="gray",
        linewidth=2
    )

    ax.tick_params(axis="both", which="major", labelsize=ticks_font_size)
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_title(title, fontsize=title_font_size)
    ax.grid(True)

    if save_path:
        # Create a separate figure for the legend
        fig_legend = plt.figure(figsize=(6, 1))
        ax_legend = fig_legend.add_subplot(111)

        # Create custom patches for the legend
        race_patch = mpatches.Patch(facecolor=race_color, edgecolor='black', label=race_label)
        sex_patch = mpatches.Patch(facecolor=sex_color, edgecolor='black', label=sex_label)

        # Add the legend to the separate axes
        ax_legend.legend(handles=[race_patch, sex_patch], fontsize=24, loc='center', frameon=False, ncol=2)
        ax_legend.axis('off')  # Turn off the axes for the legend

        fig_legend.tight_layout()
        legend_path = os.path.join(os.path.dirname(save_path), legend_filename)
        fig_legend.savefig(legend_path, bbox_inches='tight', dpi=150)
        plt.close(fig_legend)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig

def bar_plot_value_distribution(df, attribute, title, save_path=None):
    """
    Creates a bar plot of unfairness for each client, colored by which value is leading to the maximum value.
    """
    df_plot = df.copy()
    
    if 'Value' not in df_plot.columns:
        colors = '#56B4E9' # Default blue
    else:
        # Define a palette
        # 0: red, 1: blue, 2: green, 3: orange
        palette = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
        colors = [palette.get(int(float(v)), 'gray') for v in df_plot['Value']]

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by dataset name for consistent x-axis
    try:
        df_plot['dataset_int'] = df_plot['dataset'].astype(int)
        df_plot = df_plot.sort_values('dataset_int')
    except:
        df_plot = df_plot.sort_values('dataset')

    # Use plt.bar directly for more control over colors
    ax.bar(df_plot['dataset'].astype(str), df_plot[attribute], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel('Client ID', fontsize=20, labelpad=15)
    ax.set_ylabel('Dem. Disparity', fontsize=20, labelpad=15)
    
    # Ticks
    n = len(df_plot)
    if n > 50:
        ax.set_xticks(range(0, n, 10))
        ax.set_xticklabels(df_plot['dataset'].iloc[::10].astype(str), rotation=45)
    else:
        ax.set_xticks(range(n))
        ax.set_xticklabels(df_plot['dataset'].astype(str), rotation=90)
    
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='red', label='Group 0'),
        mpatches.Patch(color='blue', label='Group 1')
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
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
        
        # Check for "medium", "small", "large" nesting (Dutch Cross-Device style)
        if isinstance(data, dict):
            first_key = next(iter(data))
            if first_key in ["medium", "small", "large"] and isinstance(data[first_key], dict):
                # Flatten: take the content of "medium" as the main data
                print(f"Detected nested dataset key '{first_key}', flattening...")
                data = data[first_key]

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
        df = pd.read_csv(file_path)
        if "model" not in df.columns:
            df["model"] = "LocalModel"
        
        # Try to extract Value if missing
        if "Value" not in df.columns:
            if "value_DP_SEX" in df.columns:
                 df["Value"] = df["value_DP_SEX"].apply(lambda x: x.split("_")[-1] if isinstance(x, str) else x)
            elif "value_DP_RACE" in df.columns:
                 df["Value"] = df["value_DP_RACE"].apply(lambda x: x.split("_")[-1] if isinstance(x, str) else x)
        
        return df
    
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
        
        # Pretty Print Model Name for Titles
        display_model_name = model
        if model == "LogisticRegression":
            display_model_name = "Logistic Regression"
        elif model == "XGBoost":
            display_model_name = "XGBoost" # Already good, but being explicit
        
        # A) Difference Histogram
        try:
            diff_df = compute_differences(model_df, fl_df)
            
            # Check columns
            cols_to_check = []
            if "DP_SEX" in diff_df.columns: cols_to_check.append("DP_SEX")
            if "DP_RACE" in diff_df.columns: cols_to_check.append("DP_RACE")
            
            if cols_to_check:
                diff_df = diff_df.dropna(subset=cols_to_check)
            
            if not diff_df.empty and cols_to_check:
                plot_path = os.path.join(base_output_dir, f"diff_hist_{safe_model_name}.pdf")
                bar_plot_differences(
                    diff_df, 
                    list(diff_df["dataset"]), 
                    title=f"{display_model_name} - {fl_method_label} Unfairness Difference", 
                    y_axis="Dem. Disparity Difference",
                    save=True,
                    fig_path=plot_path
                )
                print(f"Saved Difference Histogram: {plot_path}")
            else:
                print(f"No matching data or columns for Difference Histogram for model {model}")
                
        except Exception as e:
            print(f"Error creating difference histogram for {model}: {e}")

        # B) Distribution Dots (Scatter Comparison)
        
        # Race / Marital
        if "DP_RACE" in model_df.columns and "DP_RACE" in fl_df.columns:
            try:
                plot_path = os.path.join(base_output_dir, f"scatter_race_{safe_model_name}.pdf")
                
                # Dutch specific limits
                x_limit = 0.6 if args.dataset_name and "dutch" in args.dataset_name.lower() else None
                y_limit = 0.6 if args.dataset_name and "dutch" in args.dataset_name.lower() else None

                local_client_fairness_plot(
                    df1=model_df,
                    df2=fl_df,
                    fairness_column="DP_RACE",
                    ylabel=f"{display_model_name} Dem. Disparity",
                    xlabel=f"{fl_method_label} Dem. Disparity",
                    title="MAR Unfairness Distribution" if args.dataset_name and "dutch" in args.dataset_name.lower() else "RACE Unfairness Distribution",
                    unfairness_distribution=states_unfairness,
                    legend_labels=legend_labels,
                    save_path=plot_path,
                    x_lim=x_limit,
                    y_lim=y_limit
                )
                print(f"Saved Scatter Plot (Race/MAR): {plot_path}")
            except Exception as e:
                 print(f"Error creating scatter plot (Race) for {model}: {e}")
        else:
            print(f"Skipping Race/MAR scatter plot for {model} (DP_RACE missing)")

        # Sex
        if "DP_SEX" in model_df.columns and "DP_SEX" in fl_df.columns:
            try:
                plot_path = os.path.join(base_output_dir, f"scatter_sex_{safe_model_name}.pdf")
                
                # Dutch specific limits
                x_limit = 0.3 if args.dataset_name and "dutch" in args.dataset_name.lower() else None
                y_limit = 0.3 if args.dataset_name and "dutch" in args.dataset_name.lower() else None

                local_client_fairness_plot(
                    df1=model_df,
                    df2=fl_df,
                    fairness_column="DP_SEX",
                    ylabel=f"{display_model_name} Dem. Disparity",
                    xlabel=f"{fl_method_label} Dem. Disparity",
                    title="SEX Unfairness Distribution",
                    unfairness_distribution=states_unfairness,
                    legend_labels=legend_labels,
                    save_path=plot_path,
                    x_lim=x_limit,
                    y_lim=y_limit
                )
                print(f"Saved Scatter Plot (Sex): {plot_path}")
            except Exception as e:
                 print(f"Error creating scatter plot (Sex) for {model}: {e}")
        else:
            print(f"Skipping Sex scatter plot for {model} (DP_SEX missing)")

        # Attribute Bias Distribution (Scatter DP_RACE vs DP_SEX for Local Models)
        if "DP_RACE" in model_df.columns and "DP_SEX" in model_df.columns:
            try:
                plot_path = os.path.join(base_output_dir, f"attribute_bias_dist_{safe_model_name}.pdf")
                
                scatter_fairness_plot(
                    df1=model_df,
                    client_column="dataset",
                    fairness_column_X="DP_RACE",
                    fairness_column_Y="DP_SEX",
                    ylabel="Dem. Disparity SEX",
                    xlabel="Dem. Disparity MAR", # or MARITAL for Dutch
                    title="Attribute Bias Distribution",
                    unfairness_distribution=states_unfairness,
                    legend_labels=legend_labels,
                    save_path=plot_path,
                    legend_filename=f"attribute_bias_legend_{safe_model_name}.pdf"
                )
                print(f"Saved Attribute Bias Distribution Plot: {plot_path}")
            except Exception as e:
                print(f"Error creating attribute bias distribution plot for {model}: {e}")

        # C) Value Plots (if applicable)
        # Check if we can/should generate value plots
        generate_value_plots = False
        if "Value" in fl_df.columns:
            generate_value_plots = True
            if "Value" not in model_df.columns:
                model_df_with_val = pd.merge(model_df, fl_df[["dataset", "Value"]], on="dataset", how="left")
            else:
                model_df_with_val = model_df.copy()
        
        elif args.dataset_name and "dutch" in args.dataset_name.lower() and "value" in args.experiment_name.lower():
            # Heuristic for Dutch Value: Split clients into two groups (0 and 1) based on index
            print("DEBUG: Applying Dutch Value heuristic (Value 0/1 split by index)")
            generate_value_plots = True
            model_df_with_val = model_df.copy()
            
            # Sort by dataset (assuming numeric or consistent string sort)
            # Try to convert to int for sorting
            try:
                model_df_with_val["dataset_int"] = model_df_with_val["dataset"].astype(int)
                model_df_with_val = model_df_with_val.sort_values("dataset_int")
            except:
                model_df_with_val = model_df_with_val.sort_values("dataset")
            
            num_clients = len(model_df_with_val["dataset"].unique())
            half = num_clients // 2
            
            # Create a mapping
            client_ids = model_df_with_val["dataset"].unique()
            val_map = {}
            for i, cid in enumerate(client_ids):
                val_map[cid] = 0 if i < half else 1
            
            model_df_with_val["Value"] = model_df_with_val["dataset"].map(val_map)
            
            # Add Value to fl_df as well for arrow plots
            if "Value" not in fl_df.columns:
                fl_df["Value"] = fl_df["dataset"].map(val_map)

        if generate_value_plots:
            # Ensure Value is numeric
            model_df_with_val["Value"] = pd.to_numeric(model_df_with_val["Value"], errors='coerce')
            model_df_with_val = model_df_with_val.dropna(subset=["Value"])
            
            # 1. Value Distribution (Local)
            if "DP_SEX" in model_df_with_val.columns:
                try:
                    custom_xticklabels = None
                    if args.dataset_name and "dutch" in args.dataset_name.lower():
                         custom_xticklabels = {0: "Female", 1: "Male"}

                    create_value_plot(
                        model_df_with_val, 
                        y_label="Dem. Disparity", 
                        title="Value Bias Distribution (Sex)", 
                        attribute="DP_SEX", 
                        save=True,
                        save_path=os.path.join(base_output_dir, f"value_dist_sex_{safe_model_name}.pdf"),
                        custom_labels=custom_xticklabels
                    )
                    print(f"Saved Value Distribution Plot (Sex) for {model}")
                except Exception as e:
                    print(f"Error creating value distribution plot (Sex) for {model}: {e}")

                try:
                    bar_plot_value_distribution(
                        model_df_with_val, 
                        attribute="DP_SEX", 
                        title="Value Based Distribution (Sex)", 
                        save_path=os.path.join(base_output_dir, f"value_based_dist_sex_{safe_model_name}.pdf")
                    )
                    print(f"Saved Value Based Distribution Plot (Sex) for {model}")
                except Exception as e:
                    print(f"Error creating value based distribution plot (Sex) for {model}: {e}")

            if "DP_RACE" in model_df_with_val.columns:
                try:
                    create_value_plot(
                        model_df_with_val, 
                        y_label="Dem. Disparity", 
                        title="Value Bias Distribution (Race/Mar)", 
                        attribute="DP_RACE", 
                        save=True,
                        save_path=os.path.join(base_output_dir, f"value_dist_race_{safe_model_name}.pdf")
                    )
                    print(f"Saved Value Distribution Plot (Race) for {model}")
                except Exception as e:
                    print(f"Error creating value distribution plot (Race) for {model}: {e}")

                try:
                    bar_plot_value_distribution(
                        model_df_with_val, 
                        attribute="DP_RACE", 
                        title="Value Based Distribution (Race/Mar)", 
                        save_path=os.path.join(base_output_dir, f"value_based_dist_race_{safe_model_name}.pdf")
                    )
                    print(f"Saved Value Based Distribution Plot (Race) for {model}")
                except Exception as e:
                    print(f"Error creating value based distribution plot (Race) for {model}: {e}")

            # 2. Value Change Arrows (Local -> FL)
            # Needs 'Value' in fl_df
            
            # Sex
            if "DP_SEX" in model_df_with_val.columns and "DP_SEX" in fl_df.columns:
                try:
                    custom_xticklabels = None
                    if args.dataset_name and "dutch" in args.dataset_name.lower():
                         custom_xticklabels = {0: "Female", 1: "Male"}

                    visualize_value_change(
                        df1=model_df_with_val,
                        df2=fl_df,
                        sensitive_col="DP_SEX",
                        value_col="Value",
                        title=f"Change in Max. Value Disparity (Sex)",
                        y_label="Dem. Disparity",
                        initial_state=model,
                        legend_filename=f"arrow_legend_sex_{safe_model_name}.pdf",
                        save_path=os.path.join(base_output_dir, f"arrow_sex_{safe_model_name}.pdf"),
                        custom_labels=custom_xticklabels
                    )
                    print(f"Saved Arrow Plot (Sex) for {model}")
                except Exception as e:
                    print(f"Error creating arrow plot (Sex) for {model}: {e}")
            else:
                 print(f"Skipping Arrow Plot (Sex) for {model}. Missing cols.")

            # Race
            if "DP_RACE" in model_df_with_val.columns and "DP_RACE" in fl_df.columns:
                try:
                    visualize_value_change(
                        df1=model_df_with_val,
                        df2=fl_df,
                        sensitive_col="DP_RACE",
                        value_col="Value",
                        title=f"Change in Max. Value Disparity (Race/Mar)",
                        y_label="Dem. Disparity",
                        initial_state=model,
                        legend_filename=f"arrow_legend_race_{safe_model_name}.pdf",
                        save_path=os.path.join(base_output_dir, f"arrow_race_{safe_model_name}.pdf")
                    )
                    print(f"Saved Arrow Plot (Race) for {model}")
                except Exception as e:
                    print(f"Error creating arrow plot (Race) for {model}: {e}")
            else:
                 print(f"Skipping Arrow Plot (Race) for {model}. Missing cols.")
        # D) Bias Direction Change Arrows (Dynamic X-axis based on Bias Sign)
        # This visualizes if the model flips bias against a different group (e.g. 0 to 1)
        
        # Sex
        if "DP_SEX" in model_df.columns and "DP_SEX" in fl_df.columns:
            try:
                # Create copies to avoid modifying originals for other plots
                local_dir_df = model_df.copy()
                fl_dir_df = fl_df.copy()
                
                # Define Bias Direction: 0 if DP < 0, 1 if DP >= 0
                local_dir_df["Bias_Direction"] = (local_dir_df["DP_SEX"] >= 0).astype(int)
                fl_dir_df["Bias_Direction"] = (fl_dir_df["DP_SEX"] >= 0).astype(int)
                
                custom_labels = {0: "Favors Group 0 (-)", 1: "Favors Group 1 (+)"}
                if args.dataset_name and "dutch" in args.dataset_name.lower():
                     custom_labels = {0: "Favors Female (-)", 1: "Favors Male (+)"}

                visualize_value_change(
                    df1=local_dir_df,
                    df2=fl_dir_df,
                    sensitive_col="DP_SEX",
                    value_col="Bias_Direction",
                    title=f"Change in Bias Direction (Sex)",
                    y_label="Dem. Disparity",
                    initial_state=display_model_name,
                    legend_filename=f"arrow_bias_dir_legend_sex_{safe_model_name}.pdf",
                    save_path=os.path.join(base_output_dir, f"arrow_bias_dir_sex_{safe_model_name}.pdf"),
                    custom_labels=custom_labels
                )
                print(f"Saved Bias Direction Arrow Plot (Sex) for {model}")
            except Exception as e:
                print(f"Error creating bias direction arrow plot (Sex) for {model}: {e}")

        # Race
        if "DP_RACE" in model_df.columns and "DP_RACE" in fl_df.columns:
            try:
                local_dir_df = model_df.copy()
                fl_dir_df = fl_df.copy()
                
                # Define Bias Direction: 0 if DP < 0, 1 if DP >= 0
                local_dir_df["Bias_Direction"] = (local_dir_df["DP_RACE"] >= 0).astype(int)
                fl_dir_df["Bias_Direction"] = (fl_dir_df["DP_RACE"] >= 0).astype(int)
                
                custom_labels = {0: "Favors Group 0 (-)", 1: "Favors Group 1 (+)"}

                visualize_value_change(
                    df1=local_dir_df,
                    df2=fl_dir_df,
                    sensitive_col="DP_RACE",
                    value_col="Bias_Direction",
                    title=f"Change in Bias Direction (Race/Mar)",
                    y_label="Dem. Disparity",
                    initial_state=display_model_name,
                    legend_filename=f"arrow_bias_dir_legend_race_{safe_model_name}.pdf",
                    save_path=os.path.join(base_output_dir, f"arrow_bias_dir_race_{safe_model_name}.pdf"),
                    custom_labels=custom_labels
                )
                print(f"Saved Bias Direction Arrow Plot (Race) for {model}")
            except Exception as e:
                print(f"Error creating bias direction arrow plot (Race) for {model}: {e}")
        else:
             print(f"Skipping Value Plots for {model}. 'Value' column not found in FL results.")

        # E) Dominant Bias Change Arrows (Did the primary source of bias shift?)
        # X-Axis: Sex Dominant vs Race Dominant
        # Y-Axis: Max(abs(DP_SEX), abs(DP_RACE))
        
        if "DP_SEX" in model_df.columns and "DP_RACE" in fl_df.columns and "DP_SEX" in fl_df.columns and "DP_RACE" in model_df.columns:
            try:
                local_dom_df = model_df.copy()
                fl_dom_df = fl_df.copy()
                
                # Calculate Max and Dominant for Local
                local_dom_df["Abs_Sex"] = local_dom_df["DP_SEX"].abs()
                local_dom_df["Abs_Race"] = local_dom_df["DP_RACE"].abs()
                local_dom_df["Max_Unfairness"] = local_dom_df[["Abs_Sex", "Abs_Race"]].max(axis=1)
                # 0 for Sex, 1 for Race
                local_dom_df["Dominant_Source"] = np.where(local_dom_df["Abs_Sex"] >= local_dom_df["Abs_Race"], 0, 1)
                
                # Calculate Max and Dominant for FL
                fl_dom_df["Abs_Sex"] = fl_dom_df["DP_SEX"].abs()
                fl_dom_df["Abs_Race"] = fl_dom_df["DP_RACE"].abs()
                fl_dom_df["Max_Unfairness"] = fl_dom_df[["Abs_Sex", "Abs_Race"]].max(axis=1)
                fl_dom_df["Dominant_Source"] = np.where(fl_dom_df["Abs_Sex"] >= fl_dom_df["Abs_Race"], 0, 1)
                
                custom_labels = {0: "Sex Dominant", 1: "Race/Mar Dominant"}
                
                visualize_value_change(
                    df1=local_dom_df,
                    df2=fl_dom_df,
                    sensitive_col="Max_Unfairness",
                    value_col="Dominant_Source",
                    title=f"Change in Dominant Bias Source",
                    y_label="Max Dem. Disparity",
                    initial_state=display_model_name,
                    legend_filename=f"arrow_dominant_legend_{safe_model_name}.pdf",
                    save_path=os.path.join(base_output_dir, f"arrow_dominant_bias_{safe_model_name}.pdf"),
                    custom_labels=custom_labels
                )
                print(f"Saved Dominant Bias Arrow Plot for {model}")
            except Exception as e:
                print(f"Error creating dominant bias arrow plot for {model}: {e}")

    # End of Main Loop

if __name__ == "__main__":
    main()