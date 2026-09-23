"""Plots total current, energy, and power by setup, split into subplots by
selectivity role (selective vs. classic) -- one PNG per (metric, clip),
6 files total.

Reads mcu/eval/online/<board>/pec/{current,energy,power}_results.csv.
Each has two rows per (case, model, setup): one for the "normal" clip,
one for "anomaly" (see the "role" column). Rather than average or overlay
them, each metric is rendered as two separate images, one per clip -- so
this produces 3 metrics x 2 clips = 6 PNGs per run, all into the same
pec/ folder the CSVs live in.

Naming note: this file's dataframes have a "role" COLUMN meaning clip
role (normal/anomaly). plot_timing.py/plot_footprint_highlevel.py also
use the word "role" throughout, for something unrelated -- Selective/
Classic/Unknown, computed from ckpt_descriptor.json, never read from a
column. To keep the two apart, this file always says "clip_role" for the
former and reserves bare "role" for the selectivity grouping, exactly as
in the scripts this borrows its structure from.

Unlike timing's four stages or footprint's groups, feature/backbone/
total in these CSVs are NOT additive -- total is measured independently
and already includes feature, backbone, normalize, and head. Stacking
them the way those two scripts stack their components would double-
count feature and backbone inside total. Per Tudor's call, this script
sidesteps the question entirely for now and only plots "total" -- one
bar per model per setup.

Models are distinguished by color rather than hatch pattern, unlike the
two scripts this borrows its structure from -- their hatching existed to
tell segments apart within a STACKED bar; with nothing stacked here,
color is the simpler and clearer choice.

current_results.csv stores current in uA (matches the PPK2's own unit,
and compute_current_results.py's/compute_energy_power_results.py's
column names say so explicitly). That's an awkward scale to read off a
bar chart when currents run in the hundreds of thousands, so this file
scales it to mA for display only -- the CSV itself is untouched.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import PROJECT_ROOT

# (metric key, source CSV, value column, axis/title label, display scale
# factor applied to the column before plotting -- 1.0 for columns already
# in their display unit, 0.001 for current_ua_mean -> mA)
METRICS = [
    ("power", "power_results.csv", "total_power_mw", "Power (mW)", 1.0),
    ("energy", "energy_results.csv", "total_energy_mj", "Energy (mJ)", 1.0),
    ("current", "current_results.csv", "total_current_ua_mean", "Current (mA)", 0.001),
]
CLIP_ROLES = ["normal", "anomaly"]

# Cycled across distinct config signatures -- same palette
# plot_footprint_highlevel.py uses for its groups, for visual consistency
# across mcu/eval/'s plots.
MODEL_COLOR_PALETTE = [
    "#3274a1", "#e1812c", "#c63c32", "#3a923a",
    "#9467bd", "#8c564b", "#7f7f7f", "#bcbd22",
]

CASE_COL = "case"
MODEL_COL = "model_hash"
SETUP_COL = "setup_id"
CLIP_ROLE_COL = "role"  # the clip (normal/anomaly) -- NOT the selectivity role below

ROLE_PRIORITY = {"Selective": 0, "Classic": 1, "Unknown": 2}


def load_model_config(case, model_hash, project_root):
    """Loads the "model" section of a checkpoint descriptor.

    Args:
        case: Case identifier as it appears in the results CSV, e.g. "case1".
        model_hash: Model hash as it appears in the results CSV.
        project_root: Root of the ssm-mcu-asd-ind repository.

    Returns:
        The "model" dict from ckpt_descriptor.json, or None if the
        descriptor file doesn't exist.
    """
    descriptor_path = project_root / "mcu" / "deploy" / case / model_hash / "ckpt_descriptor.json"
    if not descriptor_path.exists():
        return None
    with open(descriptor_path, "r") as f:
        return json.load(f).get("model")


def build_selectivity_groups(df, project_root):
    """Groups (case, model_hash) combinations by selectivity role.

    Args:
        df: Results already filtered to one clip role; must contain
            "case" and "model_hash" columns.
        project_root: Root of the ssm-mcu-asd-ind repository.

    Returns:
        Dict mapping (case, role) to a list of (model_hash, config_or_none)
        tuples, sorted by model_hash. role is "Selective", "Classic", or
        "Unknown".
    """
    combos = df[[CASE_COL, MODEL_COL]].drop_duplicates().itertuples(index=False)
    groups = {}
    for case, model_hash in combos:
        config = load_model_config(case, model_hash, project_root)
        role = "Unknown" if config is None else ("Selective" if config["selective"] else "Classic")
        groups.setdefault((case, role), []).append((model_hash, config))
    for key in groups:
        groups[key].sort(key=lambda m: m[0])
    return groups


def find_varying_keys(configs):
    """Finds which config fields differ across the given configs.

    Args:
        configs: A list of "model" config dicts (may be empty).

    Returns:
        A list of config keys, excluding "selective", whose value differs
        across at least two of the given configs.
    """
    if not configs:
        return []
    keys = [k for k in configs[0] if k != "selective"]
    return [k for k in keys if len({c[k] for c in configs}) > 1]


def compute_config_signature(config, varying_keys):
    """Builds a hashable signature from a model's distinguishing config values.

    Args:
        config: A model's config dict, or None if unresolved.
        varying_keys: Config fields that differ across models.

    Returns:
        A tuple of values, or None if there's no config or nothing varies.
    """
    if config is None or not varying_keys:
        return None
    return tuple(config.get(k) for k in varying_keys)


def build_color_assignment(all_members, varying_keys):
    """Assigns one color, and one vertical-offset rank, per config signature.

    Args:
        all_members: A flat list of (model_hash, config_or_none) across
            every role group.
        varying_keys: Config fields that differ across models.

    Returns:
        A tuple (color_by_hash, rank_by_hash):
            color_by_hash: dict mapping model_hash to a hex color string.
            rank_by_hash: dict mapping model_hash to an integer offset
                rank -- rank 0 sits at the top of its setup's row, in
                every subplot.
    """
    signatures = []
    for _, config in all_members:
        sig = compute_config_signature(config, varying_keys)
        if sig is not None and sig not in signatures:
            signatures.append(sig)
    rank_by_signature = {sig: i for i, sig in enumerate(signatures)}
    color_by_signature = {
        sig: MODEL_COLOR_PALETTE[rank % len(MODEL_COLOR_PALETTE)]
        for sig, rank in rank_by_signature.items()
    }

    color_by_hash = {}
    rank_by_hash = {}
    next_fallback = len(signatures)
    for model_hash, config in all_members:
        sig = compute_config_signature(config, varying_keys)
        if sig is not None:
            color_by_hash[model_hash] = color_by_signature[sig]
            rank_by_hash[model_hash] = rank_by_signature[sig]
        else:
            color_by_hash[model_hash] = MODEL_COLOR_PALETTE[next_fallback % len(MODEL_COLOR_PALETTE)]
            rank_by_hash[model_hash] = next_fallback
            next_fallback += 1
    return color_by_hash, rank_by_hash


def describe_group(case, role):
    """Builds a short subplot title for a selectivity group."""
    return f"{case} · {role} models"


def model_legend_label(model_hash, config, varying_keys):
    """Builds the legend text identifying one model's color."""
    if config is not None and varying_keys:
        cfg_str = ", ".join(f"{k}={config[k]}" for k in varying_keys if k in config)
        return f"{cfg_str} ({model_hash})"
    return model_hash


def plot_group(ax, df, case, role, members, varying_keys, setup_positions,
              color_by_hash, rank_by_hash, value_col, scale):
    """Draws all models sharing a selectivity role as one horizontal bar
    per model per setup -- not stacked, not grouped by section (see the
    module docstring for why sectioning is skipped for now).

    Args:
        ax: Matplotlib axes to draw into.
        df: Results, already filtered to one clip role.
        case: Case identifier shared by every member, e.g. "case1".
        role: "Selective", "Classic", or "Unknown".
        members: List of (model_hash, config_or_none) for this role.
        varying_keys: Config fields that differ across models.
        setup_positions: Dict mapping every setup_id to its shared
            y-axis position.
        color_by_hash: Dict mapping model_hash to its hex color.
        rank_by_hash: Dict mapping model_hash to its offset rank.
        value_col: The CSV column to plot as bar length, e.g. "total_power_mw".
        scale: Multiplier applied to value_col before plotting, to convert
            the CSV's storage unit into the display unit (e.g. uA -> mA).
    """
    model_hashes = sorted((h for h, _ in members), key=lambda h: rank_by_hash[h])
    config_by_hash = dict(members)

    n_models = len(model_hashes)
    bar_height = min(0.28, 0.8 / max(n_models, 1))
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * (bar_height * 1.2)
    offset_by_hash = dict(zip(model_hashes, offsets))

    for model_hash in model_hashes:
        model_rows = df[(df[CASE_COL] == case) & (df[MODEL_COL] == model_hash)]
        for _, row in model_rows.iterrows():
            y = setup_positions[row[SETUP_COL]] + offset_by_hash[model_hash]
            ax.barh(
                y,
                float(row[value_col]) * scale,
                height=bar_height,
                color=color_by_hash[model_hash],
                edgecolor="black",
                linewidth=0.6,
            )

    sorted_setups = sorted(setup_positions, key=setup_positions.get)
    ax.set_yticks(list(setup_positions.values()))
    ax.set_yticklabels(sorted_setups, fontsize=8)
    ax.invert_yaxis()  # Claude comment: first setup (alphabetically) at the top.

    model_legend_handles = [
        Patch(facecolor=color_by_hash[model_hash], edgecolor="black",
             label=model_legend_label(model_hash, config_by_hash[model_hash], varying_keys))
        for model_hash in model_hashes
    ]
    ax.legend(handles=model_legend_handles, title="Model", loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title(describe_group(case, role), fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def generate_total_chart(df, value_col, axis_label, scale, clip_role, output_path, project_root):
    """Renders one subplot per (case, selectivity role) group, side by
    side, for one metric and one clip.

    Args:
        df: Results already filtered to a single clip_role.
        value_col: The CSV column to plot as bar length.
        axis_label: Axis/title label, e.g. "Power (mW)".
        scale: Multiplier applied to value_col before plotting.
        clip_role: "normal" or "anomaly" -- used only in the title.
        output_path: Where to save the rendered PNG.
        project_root: Root of the ssm-mcu-asd-ind repository.
    """
    groups = build_selectivity_groups(df, project_root)
    if not groups:
        raise ValueError(f"No models found for clip_role={clip_role!r}.")

    ordered_keys = sorted(groups.keys(), key=lambda ck: (ck[0], ROLE_PRIORITY.get(ck[1], 3)))
    all_members = [member for members in groups.values() for member in members]
    varying_keys = find_varying_keys([config for _, config in all_members if config is not None])
    color_by_hash, rank_by_hash = build_color_assignment(all_members, varying_keys)

    all_setups = sorted(df[SETUP_COL].dropna().unique())
    setup_positions = {setup: i for i, setup in enumerate(all_setups)}

    n_cols = len(ordered_keys)
    fig_height = max(6, len(all_setups) * 0.32)
    fig_width = 6.5 * n_cols

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), squeeze=False, sharey=True, sharex=True)
    axes_flat = axes.flatten()

    for ax, (case, role) in zip(axes_flat, ordered_keys):
        plot_group(ax, df, case, role, groups[(case, role)], varying_keys,
                  setup_positions, color_by_hash, rank_by_hash, value_col, scale)
        ax.set_xlabel(axis_label, fontsize=10, fontweight="bold")

    axes_flat[0].set_ylabel("Setup ID", fontsize=11, fontweight="bold")
    fig.suptitle(f"Total {axis_label} by Setup, Grouped by Selectivity ({clip_role} clip)",
                fontsize=14, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot successfully written to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=str, required=True)
    args = parser.parse_args()

    board_list = ['stm32-nucleo-h7s3l8', 'esp32', 'arduino-nano-rp2040-connect-mbed',
                  'arduino-nano-rp2040-connect-pico', 'arduino-nano-rp2040-connect-pico-q15']
    if args.board not in board_list:
        raise ValueError(f"Board not supported. Must be one of {board_list}")

    pec_dir = PROJECT_ROOT / "mcu" / "eval" / "online" / args.board / "pec"

    for metric_key, csv_name, value_col, axis_label, scale in METRICS:
        df = pd.read_csv(pec_dir / csv_name)
        for clip_role in CLIP_ROLES:
            role_df = df[df[CLIP_ROLE_COL] == clip_role]
            if role_df.empty:
                print(f"No '{clip_role}' rows in {csv_name} -- skipping.")
                continue
            out_file = pec_dir / f"{metric_key}_total_{clip_role}.png"
            generate_total_chart(role_df, value_col, axis_label, scale, clip_role, out_file, PROJECT_ROOT)