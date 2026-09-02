import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runs.compute_hash import train_config_hash
from src.config import PROJECT_ROOT, load_config, load_config_by_name

metric_styles = {
    "precision": {
        "label": "Precision",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "recall": {
        "label": "Recall",
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
    },
    "accuracy": {
        "label": "Accuracy",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
    },
    "f1": {
        "label": "F1 Score",
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-",
    },
}

def run_visualize(held_out_case, cfg, dir_name):
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name
    threshold_analysis_dir = base_dir / "secondary_metrics_visuals"
    threshold_analysis_dir.mkdir(parents=True, exist_ok=True)
    threshold_metrics_path = base_dir / "thresholds_same_machine.csv"
    df = pd.read_csv(threshold_metrics_path)

    pooling_methods = df["pooling"].unique()
    distance_heads = df["distance_head"].unique()
    metrics = ["precision", "recall", "accuracy", "f1"]

    for pool in pooling_methods:
        df_pool = df[df["pooling"] == pool]

        fig, axes = plt.subplots(2, 2, figsize=(22, 16), sharey=True)
        axes = axes.flatten()

        for idx, dist_head in enumerate(distance_heads):
            ax = axes[idx]
            df_sub = df_pool[df_pool["distance_head"] == dist_head].copy()

            x_methods = df_sub["threshold_method"].tolist()
            x_indices = np.arange(len(x_methods))

            # Plot each metric directly at the tick mark without offset
            for metric in metrics:
                ax.plot(
                    x_indices,
                    df_sub[metric].values,
                    label=metric_styles[metric]["label"],
                    color=metric_styles[metric]["color"],
                    marker=metric_styles[metric]["marker"],
                    linestyle=metric_styles[metric]["linestyle"],
                    linewidth=2.2,
                    markersize=6.5,
                    alpha=0.9,
                    zorder=3,
                )

            # Formatting with 0.5 to 1.0 vertical scale
            ax.set_title(
                f"Distance Head: {dist_head}",
                fontsize=14,
                fontweight="bold",
                pad=12,
            )
            ax.set_ylim(0.0, 1.02)
            ax.set_ylabel("Score", fontsize=11)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_methods, rotation=45, ha="right", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower left", framealpha=0.9, fontsize=10)

            # Threshold value table under each subplot
            table_data = [[f"{val:.4f}" for val in df_sub["threshold"].values]]
            tbl = ax.table(
                cellText=table_data,
                rowLabels=["Threshold"],
                loc="bottom",
                bbox=[0.0, -0.48, 1.0, 0.08],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)

            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#cccccc")
                if c == -1:
                    cell.set_facecolor("#e6e6e6")
                    cell.set_text_props(weight="bold")
                else:
                    cell.set_facecolor("#f9f9f9")

        fig.suptitle(
            f"Performance Metrics & Thresholds (Pooling: {pool})",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.subplots_adjust(
            hspace=0.65,
            wspace=0.15,
            bottom=0.18,
            top=0.92,
            left=0.06,
            right=0.98,
        )

        output_filename = str(threshold_analysis_dir / f"metrics_pooling_{pool}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_filename}")

if __name__ == '__main__':
    configs_root = PROJECT_ROOT / 'configs' / 'ablation'
    runs_root = PROJECT_ROOT / 'runs'
    configs_manifest = pd.read_csv(configs_root / '000_config_manifest.csv')

    for index, row in configs_manifest.iterrows():
        case = int(row['held_out_case'])
        model_hash = row['model_hash']
        ckpt_path = runs_root / f'case{case}' / model_hash / 'ckpt.pt'

        if not ckpt_path.exists():
            print(f'model {model_hash} not cached')
            continue

        config_file = load_config_by_name(row['config_name'])

        print(f'\n=== generating embeddings for case {case} -> model {model_hash} ===')
        run_visualize(case, config_file, model_hash)
