"""Plots per-stage MCU inference latency, split into subplots by
selectivity role (selective vs. classic).

Bars are horizontal: setup_id on the y-axis, execution time (ms/frame) on
the x-axis. Both subplots share the full union of setup_ids present in the
board's timing file, so a setup that exists for only one role still gets a
row in both subplots -- empty where no data exists for that model. Models
within a role occupy a fixed vertical slot regardless of whether they have
data for a given setup, so a missing model-setup pair shows up as a blank
slot rather than being hidden by re-centering the remaining bars. Hatch
patterns -- and the vertical stacking order that goes with them -- are
assigned globally by config signature (the fields that vary across
models), so the same underlying configuration always gets the same hatch
and the same above/below position in every subplot.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import PROJECT_ROOT

# Stage metrics to stack along the horizontal axis: (column, legend label, hex color).
STAGE_COLUMNS = [
    ("feature_ms_perframe", "Feature", "#3274a1"),
    ("normalize_ms_perframe", "Normalization", "#e1812c"),
    ("backbone_ms_perframe", "Backbone", "#c63c32"),
    ("head_ms_perframe", "Head", "#3a923a"),
]

# Cycled across distinct config signatures. "" means no hatch (solid).
# Index order here doubles as the vertical stacking order: rank 0 ("",
# solid) is always drawn above rank 1 and beyond, in every subplot.
HATCH_PATTERNS = ["", "///", "xxx", "...", "\\", "++", "oo", "**"]

CASE_COL = "case"
MODEL_COL = "model_hash"
SETUP_COL = "setup_id"

ROLE_PRIORITY = {"Selective": 0, "Classic": 1, "Unknown": 2}

# Hatch lines are drawn in this color regardless of a bar's fill color.
plt.rcParams["hatch.color"] = "black"
plt.rcParams["hatch.linewidth"] = 1.1


def load_model_config(case, model_hash, project_root):
    """Loads the "model" section of a checkpoint descriptor.

    Args:
        case: Case identifier as it appears in the timing CSV, e.g. "case1".
        model_hash: Model hash as it appears in the timing CSV.
        project_root: Root of the ssm-mcu-asd-ind repository.

    Returns:
        The "model" dict from ckpt_descriptor.json, or None if the
        descriptor file doesn't exist.
    """
    descriptor_path = (
        project_root / "mcu" / "deploy" / case / model_hash / "ckpt_descriptor.json"
    )
    if not descriptor_path.exists():
        return None
    with open(descriptor_path, "r") as f:
        return json.load(f).get("model")


def build_selectivity_groups(df, project_root):
    """Groups (case, model_hash) combinations by selectivity role.

    Every selective model for a case lands in one group, every classic
    model in another, so a subplot can compare same-role models directly
    even when they don't share identical setups.

    Args:
        df: Timing results; must contain "case" and "model_hash" columns.
        project_root: Root of the ssm-mcu-asd-ind repository.

    Returns:
        Dict mapping (case, role) to a list of (model_hash, config_or_none)
        tuples, sorted by model_hash. role is "Selective", "Classic", or
        "Unknown" (when the descriptor couldn't be loaded). This list order
        is just for stable iteration -- it is NOT used for vertical
        stacking order; see build_hatch_assignment for that.
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

    Fields that are constant across every model aren't useful for telling
    models apart, so only fields that actually vary get shown.

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

    Two models with the same signature are the "same configuration" for
    plotting purposes -- they should look the same wherever they appear.

    Args:
        config: A model's config dict, or None if unresolved.
        varying_keys: Config fields that differ across models.

    Returns:
        A tuple of values, or None if there's no config or nothing varies.
    """
    if config is None or not varying_keys:
        return None
    return tuple(config.get(k) for k in varying_keys)


def build_hatch_assignment(all_members, varying_keys):
    """Assigns one hatch pattern, and one stacking rank, per config signature.

    This runs once, across every role group, before any subplot is drawn.
    The rank is what makes "hatched below solid" hold everywhere: it comes
    from signature order, never from a model_hash string, so it can't
    disagree with itself between the Selective and Classic subplots the
    way alphabetical hash order did.

    Args:
        all_members: A flat list of (model_hash, config_or_none) across
            every role group.
        varying_keys: Config fields that differ across models.

    Returns:
        A tuple (hatch_by_hash, rank_by_hash):
            hatch_by_hash: dict mapping model_hash to a hatch pattern string.
            rank_by_hash: dict mapping model_hash to an integer stacking
                rank. Rank 0 is always the signature that got the solid
                ("") hatch; higher ranks stack below it, in every subplot.
    """
    signatures = []
    for _, config in all_members:
        sig = compute_config_signature(config, varying_keys)
        if sig is not None and sig not in signatures:
            signatures.append(sig)
    rank_by_signature = {sig: i for i, sig in enumerate(signatures)}
    hatch_by_signature = {
        sig: HATCH_PATTERNS[rank % len(HATCH_PATTERNS)] for sig, rank in rank_by_signature.items()
    }

    hatch_by_hash = {}
    rank_by_hash = {}
    next_fallback = len(signatures)
    for model_hash, config in all_members:
        sig = compute_config_signature(config, varying_keys)
        if sig is not None:
            hatch_by_hash[model_hash] = hatch_by_signature[sig]
            rank_by_hash[model_hash] = rank_by_signature[sig]
        else:
            # No config (or nothing varies) to key off of -- fall back to a
            # rank/pattern reserved past the ones used by real signatures.
            hatch_by_hash[model_hash] = HATCH_PATTERNS[next_fallback % len(HATCH_PATTERNS)]
            rank_by_hash[model_hash] = next_fallback
            next_fallback += 1
    return hatch_by_hash, rank_by_hash


def describe_group(case, role):
    """Builds a short subplot title for a selectivity group.

    Args:
        case: Case identifier, e.g. "case1".
        role: "Selective", "Classic", or "Unknown".

    Returns:
        A short string, e.g. "case1 · Selective models".
    """
    return f"{case} · {role} models"


def model_legend_label(model_hash, config, varying_keys):
    """Builds the legend text identifying one model's hatch pattern.

    Args:
        model_hash: The model's hash.
        config: The model's config dict, or None if unresolved.
        varying_keys: Config fields that differ across models in the figure.

    Returns:
        A short label, e.g. "d_state=8, expand=2 (086acf0275b8)".
    """
    if config is not None and varying_keys:
        cfg_str = ", ".join(f"{k}={config[k]}" for k in varying_keys if k in config)
        return f"{cfg_str} ({model_hash})"
    return model_hash


def plot_group(ax, df, case, role, members, varying_keys, setup_positions, hatch_by_hash, rank_by_hash):
    """Draws all models sharing a selectivity role as horizontal stacked bars.

    Every model gets a fixed vertical offset within its setup's row, whether
    or not it has data for that setup -- so a missing model-setup pair is a
    blank slot, not a bar that slid over to fill the gap. Models are ordered
    by their global hatch rank (not by hash string), so the solid model is
    always drawn above the hatched one, in every subplot.

    Args:
        ax: Matplotlib axes to draw into.
        df: Timing results, already filtered to successful runs.
        case: Case identifier shared by every member, e.g. "case1".
        role: "Selective", "Classic", or "Unknown".
        members: List of (model_hash, config_or_none) for this role.
        varying_keys: Config fields that differ across models, used in the
            per-subplot model legend.
        setup_positions: Dict mapping every setup_id (across all subplots)
            to its shared y-axis position.
        hatch_by_hash: Dict mapping model_hash to its (globally assigned)
            hatch pattern.
        rank_by_hash: Dict mapping model_hash to its (globally assigned)
            stacking rank -- lower ranks are drawn above higher ones.

    Returns:
        Dict mapping stage label to bar artist, for the shared figure legend.
    """
    # Sort by global hatch rank, not by hash string, so stacking order
    # agrees with the Selective subplot no matter which hashes happen to
    # exist in this particular role.
    model_hashes = sorted((h for h, _ in members), key=lambda h: rank_by_hash[h])
    config_by_hash = dict(members)

    n_models = len(model_hashes)
    bar_height = min(0.28, 0.8 / max(n_models, 1))
    # Fixed per-model offset -- computed once, never adjusted per setup.
    # Rank 0 gets the most negative offset, which renders at the top of its
    # row (see the invert_yaxis() comment below); later ranks stack downward.
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * (bar_height * 1.2)
    offset_by_hash = dict(zip(model_hashes, offsets))

    stage_legend_handles = {}

    for model_hash in model_hashes:
        model_rows = df[(df[CASE_COL] == case) & (df[MODEL_COL] == model_hash)]
        for _, row in model_rows.iterrows():
            y = setup_positions[row[SETUP_COL]] + offset_by_hash[model_hash]
            left = 0.0
            for col_name, stage_label, color in STAGE_COLUMNS:
                val = float(row[col_name])
                bar = ax.barh(
                    y,
                    val,
                    left=left,
                    height=bar_height,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    hatch=hatch_by_hash[model_hash],
                )
                left += val
                if stage_label not in stage_legend_handles:
                    stage_legend_handles[stage_label] = bar

    sorted_setups = sorted(setup_positions, key=setup_positions.get)
    ax.set_yticks(list(setup_positions.values()))
    ax.set_yticklabels(sorted_setups, fontsize=8)
    ax.invert_yaxis()  # Claude comment: first setup (alphabetically) at the top.

    model_legend_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch_by_hash[model_hash],
            label=model_legend_label(model_hash, config_by_hash[model_hash], varying_keys),
        )
        for model_hash in model_hashes  # Same top-to-bottom order as the bars.
    ]
    ax.legend(handles=model_legend_handles, title="Model", loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Execution Time (ms / frame)", fontsize=10, fontweight="bold")
    ax.set_title(describe_group(case, role), fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    return stage_legend_handles


def generate_benchmark_chart(df, output_path, project_root):
    """Renders one subplot per (case, selectivity role) group, side by side,
    with a shared setup_id axis and a shared, locked time-in-ms axis.

    Args:
        df: Timing results loaded from a board's timing_results.csv.
        output_path: Where to save the rendered PNG.
        project_root: Root of the ssm-mcu-asd-ind repository, used to
            resolve each model's ckpt_descriptor.json.
    """
    if "status" in df.columns:
        valid_df = df[df["status"].astype(str).str.upper() == "OK"]
        if not valid_df.empty:
            df = valid_df

    groups = build_selectivity_groups(df, project_root)
    if not groups:
        raise ValueError("No models found in the provided timing data.")

    ordered_keys = sorted(groups.keys(), key=lambda ck: (ck[0], ROLE_PRIORITY.get(ck[1], 3)))
    all_members = [member for members in groups.values() for member in members]
    varying_keys = find_varying_keys([config for _, config in all_members if config is not None])
    hatch_by_hash, rank_by_hash = build_hatch_assignment(all_members, varying_keys)

    # Every setup_id in the (filtered) file, shared across every subplot.
    all_setups = sorted(df[SETUP_COL].dropna().unique())
    setup_positions = {setup: i for i, setup in enumerate(all_setups)}

    n_cols = len(ordered_keys)
    fig_height = max(8, len(all_setups) * 0.32)
    fig_width = 7.5 * n_cols

    fig, axes = plt.subplots(
        1, n_cols, figsize=(fig_width, fig_height), squeeze=False, sharey=True, sharex=True
    )
    axes_flat = axes.flatten()

    all_stage_handles = {}
    for ax, (case, role) in zip(axes_flat, ordered_keys):
        handles = plot_group(
            ax, df, case, role, groups[(case, role)], varying_keys,
            setup_positions, hatch_by_hash, rank_by_hash,
        )
        all_stage_handles.update(handles)

    axes_flat[0].set_ylabel("Setup ID", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Per-Stage Frame Latency by Setup, Grouped by Selectivity",
        fontsize=14,
        fontweight="bold",
    )
    fig.legend(
        all_stage_handles.values(),
        all_stage_handles.keys(),
        title="Computation Stage",
        loc="upper center",
        ncol=len(all_stage_handles),
        bbox_to_anchor=(0.80, 0.95),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot successfully written to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=str, required=True)
    args = parser.parse_args()

    board_list = ["stm32-nucleo-h7s3l8", "esp32", "arduino-uno-rp2040"]
    if args.board not in board_list:
        raise ValueError(f"Board not supported. Must be one of {board_list}")

    timing_file = PROJECT_ROOT / "mcu" / "eval" / "online" / args.board / "timing_results.csv"
    timing_df = pd.read_csv(timing_file)
    out_file = (
        PROJECT_ROOT / "mcu" / "eval" / "online" / args.board / "timing" / "latency_breakdown.png"
    )
    generate_benchmark_chart(timing_df, out_file, PROJECT_ROOT)