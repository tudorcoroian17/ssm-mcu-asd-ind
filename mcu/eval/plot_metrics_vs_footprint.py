#!/usr/bin/env python3
import argparse
import os
import re
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# 1. Quantization Degradation Ranking Lists
# (Ordered from Mildest / Least Quantized -> Most Aggressive / Darkest Shade)
# You can reorder the items in these 4 lists as needed.
# ---------------------------------------------------------
RANK_SELECTIVE_EUCLIDEAN = [
    "w_proj_perchannel_euclidean_fp32",
    "w_proj_pertensor_euclidean_fp32",
    "w_all_perchannel_euclidean_fp32",
    "w_all_pertensor_euclidean_fp32",
    "w_all_pertensor_actboundaries_euclidean_fp32",
    "true_int8_h16_euclidean_fp32",
    "true_int8_h16_euclidean_int8",
    "true_int8_h8_euclidean_fp32",
    "true_int8_h8_euclidean_int8",
]

RANK_SELECTIVE_KNN16 = [
    "w_proj_perchannel_knn16_fp32",
    "w_proj_pertensor_knn16_fp32",
    "w_all_perchannel_knn16_fp32",
    "w_all_pertensor_knn16_fp32",
    "w_all_pertensor_actboundaries_knn16_fp32",
    "true_int8_h16_knn16_fp32",
    "true_int8_h16_knn16_int8",
    "true_int8_h8_knn16_fp32",
    "true_int8_h8_knn16_int8",
]

RANK_CLASSIC_EUCLIDEAN = [
    "w_all_perchannel_qab_euclidean_fp32",
    "w_all_perchannel_qabar_euclidean_fp32",
    "w_all_pertensor_qab_euclidean_fp32",
    "w_all_pertensor_qabar_euclidean_fp32",
    "w_all_pertensor_actboundaries_qab_euclidean_fp32",
    "w_all_pertensor_actboundaries_qabar_euclidean_fp32",
    "true_int8_h16_qabar_euclidean_fp32",
    "true_int8_h16_qab_euclidean_fp32",
    "true_int8_h16_qabar_euclidean_int8",
    "true_int8_h16_qab_euclidean_int8",
    "true_int8_h8_qabar_euclidean_fp32",
    "true_int8_h8_qab_euclidean_fp32",
    "true_int8_h8_qabar_euclidean_int8",
    "true_int8_h8_qab_euclidean_int8",
]

RANK_CLASSIC_KNN16 = [
    "w_all_perchannel_qab_knn16_fp32",
    "w_all_perchannel_qabar_knn16_fp32",
    "w_all_pertensor_qab_knn16_fp32",
    "w_all_pertensor_qabar_knn16_fp32",
    "w_all_pertensor_actboundaries_qab_knn16_fp32",
    "w_all_pertensor_actboundaries_qabar_knn16_fp32",
    "true_int8_h16_qabar_knn16_fp32",
    "true_int8_h16_qab_knn16_fp32",
    "true_int8_h16_qabar_knn16_int8",
    "true_int8_h16_qab_knn16_int8",
    "true_int8_h8_qabar_knn16_fp32",
    "true_int8_h8_qab_knn16_fp32",
    "true_int8_h8_qabar_knn16_int8",
    "true_int8_h8_qab_knn16_int8",
]

GROUPS = {
    "sel_euclidean": {
        "name": "Selective - Euclidean",
        "marker": "s",
        "cmap": cm.Blues,
        "base_color": "#1f77b4",
        "rank_list": RANK_SELECTIVE_EUCLIDEAN,
        "prefix": "",
    },
    "sel_knn16": {
        "name": "Selective - kNN16",
        "marker": "s",
        "cmap": cm.Reds,
        "base_color": "#d62728",
        "rank_list": RANK_SELECTIVE_KNN16,
        "prefix": "",
    },
    "cls_euclidean": {
        "name": "Classic - Euclidean",
        "marker": "^",
        "cmap": cm.Greens,
        "base_color": "#2ca02c",
        "rank_list": RANK_CLASSIC_EUCLIDEAN,
        "prefix": "",
    },
    "cls_knn16": {
        "name": "Classic - kNN16",
        "marker": "^",
        "cmap": cm.Oranges,
        "base_color": "#ff7f0e",
        "rank_list": RANK_CLASSIC_KNN16,
        "prefix": "",
    },
}

MODEL_PAIRS = [
    {
        "name": "pair1_f4cd_f257",
        "title": "Pair 1: f4cd557b7e3b (Selective, Pareto Winner) vs f2578cb06991 (Classic)",
        "selective_config": "f4cd557b7e3b",
        "classic_config": "f2578cb06991",
    },
    {
        "name": "pair2_b397_352f",
        "title": "Pair 2: b39731b66741 (Selective) vs 352f70960ed3 (Classic, Pareto Winner)",
        "selective_config": "b39731b66741",
        "classic_config": "352f70960ed3",
    },
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate MCU Footprint vs Performance Metric charts."
    )
    parser.add_argument(
        "--first-metrics",
        type=str,
        default="first_metrics.csv",
        help="Path to first_metrics.csv or its directory.",
    )
    parser.add_argument(
        "--second-metrics",
        type=str,
        default="second_metrics.csv",
        help="Path to second_metrics.csv or its directory.",
    )
    parser.add_argument(
        "--footprint",
        type=str,
        default="footprint_summary_stm32-nucleo-h7s3l8_old2.csv",
        help="Path to MCU footprint CSV or its directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metrics_vs_footprint_charts",
        help="Directory where generated charts will be saved.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        choices=["first", "second", "all"],
        default="all",
        help="Which metrics to plot: 'first', 'second', or 'all'.",
    )
    return parser.parse_args()


def resolve_file_path(path_str: str, default_filename: str) -> str:
    if os.path.isdir(path_str):
        candidate = os.path.join(path_str, default_filename)
        if os.path.exists(candidate):
            return candidate
        for fname in os.listdir(path_str):
            if fname.endswith(".csv") and default_filename.split(".")[0] in fname:
                return os.path.join(path_str, fname)
    return path_str


# ---------------------------------------------------------
# 2. Data Loading & Preparation
# ---------------------------------------------------------
def load_and_prepare_data(
    first_path: str, second_path: str, footprint_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_footprint = pd.read_csv(footprint_path)
    df_footprint["flash_kb"] = df_footprint["flash_bytes"] / 1024.0
    df_footprint["ram_kb"] = df_footprint["ram_bytes"] / 1024.0

    # First Metrics (AUC / pAUC)
    df_first = pd.read_csv(first_path)
    df_first["clean_config"] = (
        df_first["config"].astype(str).str.replace(".yaml", "", regex=False)
    )
    merged_first = pd.merge(
        df_first,
        df_footprint[["model", "setup", "flash_kb", "ram_kb"]],
        left_on=["model_hash", "identifier"],
        right_on=["model", "setup"],
        how="inner",
    )

    # Second Metrics (Precision, Recall, Accuracy, F1)
    df_second = pd.read_csv(second_path)
    df_second["clean_config"] = (
        df_second["config"].astype(str).str.replace(".yaml", "", regex=False)
    )
    merged_second = pd.merge(
        df_second,
        df_footprint[["model", "setup", "flash_kb", "ram_kb"]],
        left_on=["model_hash", "identifier"],
        right_on=["model", "setup"],
        how="inner",
    )

    return merged_first, merged_second


def assign_group_and_colors(df_pair: pd.DataFrame, sel_cfg: str) -> pd.DataFrame:
    rows = []
    for _, row in df_pair.iterrows():
        ident = row["identifier"]
        is_sel = row["clean_config"] == sel_cfg
        head = row["head"]

        if is_sel and head == "euclidean":
            g_key = "sel_euclidean"
        elif is_sel and head == "knn16":
            g_key = "sel_knn16"
        elif (not is_sel) and head == "euclidean":
            g_key = "cls_euclidean"
        else:
            g_key = "cls_knn16"

        grp = GROUPS[g_key]
        r_list = grp["rank_list"]
        rank_idx = r_list.index(ident) if ident in r_list else 0
        total_items = max(len(r_list) - 1, 1)

        # Normalized gradient from 0.35 (mild) to 0.95 (aggressive)
        norm_val = 0.35 + 0.60 * (rank_idx / total_items)
        color = grp["cmap"](norm_val)

        row_dict = row.to_dict()
        row_dict["group_key"] = g_key
        row_dict["rank_idx"] = rank_idx
        row_dict["color"] = color
        row_dict["marker"] = grp["marker"]
        row_dict["combo_label"] = f"{row['combo_number']}{grp['prefix']}"
        rows.append(row_dict)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. Plotting & Legend Helpers
# ---------------------------------------------------------
def render_side_legend(ax, df_plot: pd.DataFrame):
    """Renders the degradation ranking reference on the left panel."""
    ax.axis("off")
    leg_lines = ["DEGRADATION RANKING (Light -> Dark):\n"]

    for g_key, grp in GROUPS.items():
        leg_lines.append(f"[{grp['name']}]:")
        sub_g = (
            df_plot[df_plot["group_key"] == g_key]
            .drop_duplicates(subset=["identifier"])
            .sort_values("rank_idx")
        )
        for _, r in sub_g.iterrows():
            leg_lines.append(
                f"  #{r['rank_idx']+1:02d} ({r['combo_label']}) - {r['identifier']}"
            )
        leg_lines.append("")

    ax.text(
        0.0,
        0.98,
        "\n".join(leg_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fbfbfb", edgecolor="#cccccc"),
    )


def create_subplot_handles():
    """Generates standard legend handles for individual subplot legends."""
    return [
        mlines.Line2D(
            [],
            [],
            color="#1f77b4",
            marker="s",
            linestyle="None",
            markersize=6,
            label="Selective - Euclidean (Blue)",
        ),
        mlines.Line2D(
            [],
            [],
            color="#d62728",
            marker="s",
            linestyle="None",
            markersize=6,
            label="Selective - kNN16 (Red)",
        ),
        mlines.Line2D(
            [],
            [],
            color="#2ca02c",
            marker="^",
            linestyle="None",
            markersize=6,
            label="Classic - Euclidean (Green)",
        ),
        mlines.Line2D(
            [],
            [],
            color="#ff7f0e",
            marker="^",
            linestyle="None",
            markersize=6,
            label="Classic - kNN16 (Orange)",
        ),
    ]


def plot_scatter_with_bottom_legend(
    ax, df_plot: pd.DataFrame, x_col: str, y_metric: str, xlabel: str, ylabel: str
):
    for g_key, grp in GROUPS.items():
        sub = df_plot[df_plot["group_key"] == g_key]
        ax.scatter(
            sub[x_col],
            sub[y_metric],
            marker=grp["marker"],
            c=sub["color"],
            s=50,
            edgecolors="black",
            linewidth=0.7,
            alpha=0.9,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["combo_label"],
                (r[x_col], r[y_metric]),
                fontsize=7,
                color="#222222",
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_title(f"{ylabel} vs {xlabel}", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Attach legend directly under this subplot
    ax.legend(
        handles=create_subplot_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=8,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#dddddd",
    )


# ---------------------------------------------------------
# 4. Chart Generation Workflows
# ---------------------------------------------------------
def generate_first_metrics(df_first: pd.DataFrame, output_dir: str):
    out_path = os.path.join(output_dir, "first_metrics")
    os.makedirs(out_path, exist_ok=True)

    for pair in MODEL_PAIRS:
        df_pair = df_first[
            df_first["clean_config"].isin(
                [pair["selective_config"], pair["classic_config"]]
            )
        ]
        if df_pair.empty:
            continue

        df_plot = assign_group_and_colors(df_pair, pair["selective_config"])

        fig = plt.figure(figsize=(20, 11))
        gs = gridspec.GridSpec(
            2, 3, width_ratios=[0.9, 1.15, 1.15], figure=fig, wspace=0.25, hspace=0.48
        )

        # Left degradation legend
        ax_leg = fig.add_subplot(gs[:, 0])
        render_side_legend(ax_leg, df_plot)

        plots = [
            (fig.add_subplot(gs[0, 1]), "flash_kb", "auc", "Flash (KB)", "AUC"),
            (fig.add_subplot(gs[0, 2]), "ram_kb", "auc", "RAM (KB)", "AUC"),
            (fig.add_subplot(gs[1, 1]), "flash_kb", "pauc", "Flash (KB)", "pAUC"),
            (fig.add_subplot(gs[1, 2]), "ram_kb", "pauc", "RAM (KB)", "pAUC"),
        ]

        for ax, x_col, y_col, xlabel, ylabel in plots:
            plot_scatter_with_bottom_legend(ax, df_plot, x_col, y_col, xlabel, ylabel)

        fig.suptitle(pair["title"], fontsize=13, fontweight="bold", y=0.98)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)

        target_file = os.path.join(out_path, f"{pair['name']}_auc_pauc.png")
        plt.savefig(target_file, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"[Done] Generated: {target_file}")


def generate_second_metrics(df_second: pd.DataFrame, output_dir: str):
    out_path = os.path.join(output_dir, "second_metrics")
    os.makedirs(out_path, exist_ok=True)

    metrics = ["precision", "recall", "accuracy", "f1"]
    threshold_methods = sorted(df_second["threshold_method"].unique())

    for pair in MODEL_PAIRS:
        df_pair_full = df_second[
            df_second["clean_config"].isin(
                [pair["selective_config"], pair["classic_config"]]
            )
        ]
        if df_pair_full.empty:
            continue

        for tm in threshold_methods:
            df_tm = df_pair_full[df_pair_full["threshold_method"] == tm]
            if df_tm.empty:
                continue

            df_plot = assign_group_and_colors(df_tm, pair["selective_config"])

            fig = plt.figure(figsize=(20, 18))
            gs = gridspec.GridSpec(
                4, 3, width_ratios=[0.9, 1.15, 1.15], figure=fig, wspace=0.25, hspace=0.48
            )

            # Left degradation legend
            ax_leg = fig.add_subplot(gs[:, 0])
            render_side_legend(ax_leg, df_plot)

            for row_idx, metric in enumerate(metrics):
                for col_idx, (x_col, xlabel) in enumerate(
                    [("flash_kb", "Flash (KB)"), ("ram_kb", "RAM (KB)")]
                ):
                    ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                    plot_scatter_with_bottom_legend(
                        ax, df_plot, x_col, metric, xlabel, metric.capitalize()
                    )

            fig.suptitle(
                f"{pair['title']} | Threshold: {tm}",
                fontsize=13,
                fontweight="bold",
                y=0.99,
            )
            plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

            safe_tm = re.sub(r"[^\w\-.]", "_", tm)
            target_file = os.path.join(out_path, f"{pair['name']}_{safe_tm}.png")
            plt.savefig(target_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Done] Generated: {target_file}")


# ---------------------------------------------------------
# 5. Main Execution Entry
# ---------------------------------------------------------
def main():
    args = parse_arguments()

    first_file = resolve_file_path(args.first_metrics, "first_metrics.csv")
    second_file = resolve_file_path(args.second_metrics, "second_metrics.csv")
    footprint_file = resolve_file_path(
        args.footprint, "footprint_summary_stm32-nucleo-h7s3l8_old2.csv"
    )

    print(f"Loading data from:\n  - {first_file}\n  - {second_file}\n  - {footprint_file}")
    df_first, df_second = load_and_prepare_data(
        first_file, second_file, footprint_file
    )

    if args.metrics in ["first", "all"]:
        print("\n--- Generating First Metric Charts (AUC / pAUC) ---")
        generate_first_metrics(df_first, args.output_dir)

    if args.metrics in ["second", "all"]:
        print("\n--- Generating Second Metric Charts (8 Subplots x 28 Images) ---")
        generate_second_metrics(df_second, args.output_dir)

    print("\nAll requested charts generated successfully.")


if __name__ == "__main__":
    main()