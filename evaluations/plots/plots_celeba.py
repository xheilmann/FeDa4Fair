from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global Styles
FONT_SIZE_TITLE = 24
FONT_SIZE_LABELS = 22
FONT_SIZE_TICKS = 20
FONT_SIZE_LEGEND = 18


def save_legend_separately(ax, output_dir, filename):
    """
    Extracts the legend from an existing axis and saves it as a separate PDF.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    fig_legend = plt.figure(figsize=(10, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=len(handles),
        fontsize=FONT_SIZE_TICKS,
        frameon=False,
    )
    ax_legend.axis("off")
    fig_legend.tight_layout()
    legend_path = Path(output_dir) / filename
    fig_legend.savefig(legend_path, bbox_inches="tight", dpi=150)
    plt.close(fig_legend)


def plot_attribute_unfairness(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Grouped Bar Plot: Male vs Hair Color Unfairness per client.
    Saves the plot and a separate legend.
    """
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Filter/Select columns
    # Assuming CSV has 'Client ID', 'DP_Male', 'DP_Hair_Color'
    df_plot = df[["Client ID", "DP_Male", "DP_Hair_Color"]]
    df_melted = df_plot.melt(id_vars="Client ID", var_name="Attribute", value_name="Unfairness")

    _, ax = plt.subplots(figsize=(14, 6))

    # Colors
    palette = {"DP_Male": "#1E88E5", "DP_Hair_Color": "#D81B60"}

    sns.barplot(
        x="Client ID",
        y="Unfairness",
        hue="Attribute",
        data=df_melted,
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(f"Attribute Unfairness {title_suffix}", fontsize=FONT_SIZE_TITLE, pad=20)
    ax.set_xlabel("Client ID", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Demographic Disparity", fontsize=FONT_SIZE_LABELS)

    # Ticks logic
    unique_clients = df_plot["Client ID"].unique()
    n = len(unique_clients)
    max_clients_full_ticks = 50
    if n > max_clients_full_ticks:
        ticks = range(0, n, 10)
        ax.set_xticks(ticks)
    else:
        ax.set_xticks(range(n))

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

    # Legend Handling
    legend = ax.get_legend()
    if legend:
        legend.remove()
    save_legend_separately(ax, output_dir, f"{file_prefix}_attr_legend.pdf")

    # Save Plot
    output_path = Path(output_dir) / f"{file_prefix}_bar.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_attribute_scatter(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Scatter plot: Male Unfairness (X) vs Hair Color Unfairness (Y).
    """
    if not Path(csv_path).exists():
        return

    df = pd.read_csv(csv_path)

    _, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        df["DP_Male"],
        df["DP_Hair_Color"],
        s=150,
        alpha=0.7,
        edgecolors="black",
        color="#56B4E9",
    )

    # Diagonal
    max_val = max(df["DP_Male"].max(), df["DP_Hair_Color"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="gray", alpha=0.7)

    ax.set_title(f"Male vs Hair Color {title_suffix}", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("DP (Male)", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("DP (Hair Color)", fontsize=FONT_SIZE_LABELS)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.grid(visible=True)

    output_path = Path(output_dir) / f"{file_prefix}_scatter.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_value_lollipop_sorted(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Lollipop plot showing max unfairness per client, sorted by value.
    Cleaner than bar/scatter.
    """
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Assume columns: 'Client ID', 'Value', 'Max_DP'
    # 'Value' is the group index (0, 1, 2, 3...)
    df = df.sort_values(by=["Value", "Max_DP"])

    _, ax = plt.subplots(figsize=(14, 6))

    # Colors for values
    val_palette = {0: "#E69F00", 1: "#56B4E9", 2: "#009E73", 3: "#F0E442"}
    colors = [val_palette.get(v, "gray") for v in df["Value"]]

    x = range(len(df))

    # Draw Heads
    ax.scatter(x, df["Max_DP"], color=colors, s=100, zorder=3, edgecolors="black", linewidth=0.5)

    # Draw Lines (stem)
    # We color the stems gray or black for simplicity, or match the head color.
    # Matching head color looks better.
    for i, row in df.iterrows():
        xi = df.index.get_loc(i)
        ax.vlines(
            x=xi,
            ymin=0,
            ymax=row["Max_DP"],
            color=val_palette.get(row["Value"], "gray"),
            alpha=0.5,
            linewidth=1.5,
        )

    ax.set_title(f"Max Value Unfairness {title_suffix}", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("Client (Sorted by Sensitive Value)", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Max DP", fontsize=FONT_SIZE_LABELS)

    # Hide X ticks if too many
    max_clients_full_ticks_lollipop = 50
    if len(df) > max_clients_full_ticks_lollipop:
        ax.set_xticks([])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(df["Client ID"], rotation=90, fontsize=10)

    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    output_path = Path(output_dir) / f"{file_prefix}_lollipop.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_value_boxplot_grouped(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Shows the distribution of max unfairness *for clients dominated by that value*.
    """
    if not Path(csv_path).exists():
        return

    df = pd.read_csv(csv_path)

    _, ax = plt.subplots(figsize=(8, 6))

    # Seaborn Boxplot
    sns.boxplot(x="Value", y="Max_DP", data=df, ax=ax, palette="Set2")
    sns.stripplot(x="Value", y="Max_DP", data=df, ax=ax, color="black", alpha=0.3, jitter=True)

    ax.set_title(f"Distribution of Max DP per Value {title_suffix}", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("Sensitive Group Value", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Max DP", fontsize=FONT_SIZE_LABELS)

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.set_ylim(bottom=0)

    output_path = Path(output_dir) / f"{file_prefix}_boxplot.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_value_scatter_distribution(csv_path, output_dir, file_prefix, title_suffix=""):
    """
    Shows distribution of unfairness per value group (ALL unfairness values, not just max).
    """
    if not Path(csv_path).exists():
        return

    df = pd.read_csv(csv_path)

    # Expecting columns like 'DP_Val_0', 'DP_Val_1', etc.
    # Melt them
    val_cols = [c for c in df.columns if c.startswith("DP_Val_")]
    data_points = []
    for col in val_cols:
        val = int(col.split("_")[-1])
        # Only include if value was present in client
        # (Assuming -1 or NaN if missing)
        # PERF401: Replace loop with list.extend or similar
        subset = df[df[col] != -1][col]
        data_points.extend([{"Value": val, "DP": dp_val} for dp_val in subset])

    if not data_points:
        return

    df_plot = pd.DataFrame(data_points)

    _, ax = plt.subplots(figsize=(10, 6))

    rng = np.random.default_rng()
    jitter_amount = 0.15
    unique_vals = sorted(df_plot["Value"].unique())

    for val in unique_vals:
        subset = df_plot[df_plot["Value"] == val]
        y_vals = subset["DP"]
        x_base = int(val)
        x_jittered = rng.uniform(x_base - jitter_amount, x_base + jitter_amount, len(y_vals))

        ax.scatter(
            x_jittered,
            y_vals,
            alpha=0.5,
            s=80,
            edgecolors="black",
            linewidth=0.5,
            label=f"Group {val}" if val == unique_vals[0] else "",
        )

    ax.set_title(f"Unfairness Distribution per Value {title_suffix}", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("Sensitive Group Value", fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel("Demographic Disparity", fontsize=FONT_SIZE_LABELS)

    ax.set_xticks(unique_vals)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.set_ylim(bottom=0)
    ax.grid(visible=True)

    output_path = Path(output_dir) / f"{file_prefix}_scatter_dist.pdf"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    base_dir = "evaluations/results/celeba"
    output_root = "evaluations/plots/celeba_plots"

    # 1. Attribute Plots
    attr_out_dir = Path(output_root) / "attribute"
    attr_out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-Device Attribute
    csv_path = Path(base_dir) / "cross_device_attribute" / "medium_evaluation.csv"
    plot_attribute_unfairness(csv_path, attr_out_dir, "cross_device", "(Cross-Device)")
    plot_attribute_scatter(csv_path, attr_out_dir, "cross_device", "(Cross-Device)")

    # Cross-Silo Attribute
    csv_path = Path(base_dir) / "cross_silo_attribute" / "medium_evaluation.csv"
    plot_attribute_unfairness(csv_path, attr_out_dir, "cross_silo", "(Cross-Silo)")
    plot_attribute_scatter(csv_path, attr_out_dir, "cross_silo", "(Cross-Silo)")

    # 2. Value Plots
    val_out_dir = Path(output_root) / "value"
    val_out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-Device Value
    csv_path = Path(base_dir) / "cross_device_value" / "medium_evaluation.csv"
    plot_value_lollipop_sorted(csv_path, val_out_dir, "cross_device", "(Cross-Device)")
    plot_value_boxplot_grouped(csv_path, val_out_dir, "cross_device", "(Cross-Device)")
    plot_value_scatter_distribution(csv_path, val_out_dir, "cross_device", "(Cross-Device)")

    # Cross-Silo Value
    csv_path = Path(base_dir) / "cross_silo_value" / "medium_evaluation.csv"
    plot_value_lollipop_sorted(csv_path, val_out_dir, "cross_silo", "(Cross-Silo)")
    plot_value_boxplot_grouped(csv_path, val_out_dir, "cross_silo", "(Cross-Silo)")
    plot_value_scatter_distribution(csv_path, val_out_dir, "cross_silo", "(Cross-Silo)")


if __name__ == "__main__":
    main()
