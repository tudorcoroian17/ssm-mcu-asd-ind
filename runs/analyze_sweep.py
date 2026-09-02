"""
Builds the Phase 2 comparison figures from runs/phase2/.

Reads only the collected tables, never the run directories. Every figure writes
the exact rows behind it as a CSV alongside the PNG, so no plotted value is
unreachable.

Nothing here averages AUC across configs. The factorial gives something better:
every config has partners differing in exactly one knob and identical in the
other four, so a knob's effect is a set of exact run-minus-run differences.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

PHASE2 = PROJECT_ROOT / 'runs' / 'phase2'
KNOBS = ['d_state', 'n_layers', 'expand', 'selective', 'discretization']

# Contrasts read against configs/default.yaml, so every delta answers "what does
# the cheaper or simpler option cost relative to what Phase 1 shipped".
BASELINES = {'d_state': 32, 'n_layers': 4, 'expand': 2,
             'selective': 'False', 'discretization': 'zoh'}

POOLINGS = ['mean', 'max', 'concat_mean_last']
HEADS = ['euclidean', 'mahalanobis', 'knn_full', 'knn_clustered_16']

SEED_SD = 0.0132   # case1 seed spread at the default config, findings/130 s8
RP2040_FLASH_KB = 2048
RP2040_RAM_KB = 264

# findings/130 s2.4: condition numbers 6.6e6 to 1.3e8, below-chance AUC at two
# of three seeds. Shown in the grids, never reasoned from.
DISQUALIFIED = [('concat_mean_last', 'mahalanobis')]

def normalise_knobs(df):
    """
    Forces the knob columns to a stable dtype before any pivot.

    `selective` arrives as bool, uint8, or the strings 'True'/'False' depending
    on how the manifest was written and how read_csv inferred it. A pivot on a
    uint8 column produces a uint8 index, and looking that up with a Python bool
    raises KeyError. Strings dodge every numeric-index surprise and print
    readably in the contrast labels.
    """
    for knob in KNOBS:
        if knob in df:
            df[knob] = df[knob].astype(str)
    return df


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def one_cell(scores, case, pooling, head):
    """One row per config. AUC and pAUC repeat across the 14 threshold methods."""
    sub = scores[(scores.held_out_case == case)
                 & (scores.pooling == pooling)
                 & (scores.distance_head == head)]
    return sub.drop_duplicates(subset=['config_name'])


def paired_deltas(sub, knob, metric='auc'):
    """
    Exact matched differences for one knob.

    Pivoting on the other four knobs puts configs differing only in `knob` on
    the same row, so each delta is one real run minus one real run with
    everything else fixed. Uses pivot, not pivot_table: pivot raises on
    duplicate index/column pairs rather than silently averaging them.
    """
    others = [k for k in KNOBS if k != knob]
    empty = pd.DataFrame(columns=['knob', 'contrast', 'delta', 'n_pairs'])
    if sub.empty:
        return empty

    wide = sub.pivot(index=others, columns=knob, values=metric)
    base = BASELINES[knob]
    if base not in wide.columns:
        return empty

    out = []
    for level in sorted(wide.columns, key=str):
        if level == base:
            continue
        # .loc with an explicit column axis, not wide[[base, level]]: the
        # `selective` knob's levels are True and False, and a list of bools in
        # __getitem__ is read as a row mask rather than as column labels.
        paired = wide.loc[:, [base, level]].dropna()
        if paired.empty:
            continue
        out.append(pd.DataFrame({
            'knob': knob,
            'contrast': f'{level} vs {base}',
            'delta': (paired[level] - paired[base]).values,
            'n_pairs': len(paired),
        }))
    return pd.concat(out, ignore_index=True) if out else empty


def pareto_front(df, x, y):
    """Non-dominated set: minimise x, maximise y."""
    ordered = df.sort_values([x, y], ascending=[True, False])
    keep, best = [], -np.inf
    for idx, row in ordered.iterrows():
        if row[y] > best:
            keep.append(idx)
            best = row[y]
    return df.loc[keep].sort_values(x)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_paired_deltas(scores, args, outdir):
    """Figure 1. Which knob moves the metric, and how consistently."""
    sub = one_cell(scores, args.case, args.pooling, args.head)
    frames = [paired_deltas(sub, k, args.metric) for k in KNOBS]
    all_d = pd.concat(frames, ignore_index=True)
    if all_d.empty:
        print('fig1: no matched pairs yet, skipping')
        return
    all_d.to_csv(outdir / 'fig1_paired_deltas.csv', index=False)

    contrasts = list(all_d['contrast'].unique())
    rng = np.random.default_rng(0)   # jitter only; fixed so the figure is stable
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(contrasts) + 2.5))

    for i, contrast in enumerate(contrasts):
        d = all_d[all_d.contrast == contrast]['delta'].values
        ax.scatter(d, i + rng.uniform(-0.12, 0.12, len(d)),
                   s=18, alpha=0.6, color='#2a78d6')
        ax.scatter([np.median(d)], [i], marker='|', s=400,
                   color='#c0392b', zorder=3)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvspan(-SEED_SD, SEED_SD, color='grey', alpha=0.15,
               label=f'seed noise (+/-{SEED_SD})')
    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels([f"{c}  (n={all_d[all_d.contrast == c]['n_pairs'].iloc[0]})"
                        for c in contrasts])
    ax.set_xlabel(f'paired change in {args.metric}')
    ax.set_title(f'case{args.case}  {args.pooling} + {args.head}\n'
                 f'red tick = median of exact differences')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / 'fig1_paired_deltas.png', dpi=150)
    plt.close(fig)


def fig_sign_agreement(scores, args, outdir):
    """Figure 2. Does each knob's direction survive every pooling x head cell."""
    rows = []
    for pooling in POOLINGS:
        for head in HEADS:
            sub = one_cell(scores, args.case, pooling, head)
            for knob in KNOBS:
                d = paired_deltas(sub, knob, args.metric)
                for contrast, g in d.groupby('contrast'):
                    rows.append({
                        'cell': f'{pooling}\n{head}', 'contrast': contrast,
                        'n_positive': int((g.delta > 0).sum()), 'n_pairs': len(g),
                        'disqualified': (pooling, head) in DISQUALIFIED,
                    })
    agree = pd.DataFrame(rows)
    if agree.empty:
        print('fig2: no matched pairs yet, skipping')
        return
    agree.to_csv(outdir / 'fig2_sign_agreement.csv', index=False)

    agree['frac'] = agree.n_positive / agree.n_pairs
    grid = agree.pivot(index='contrast', columns='cell', values='frac')
    labels = agree.pivot(index='contrast', columns='cell', values='n_positive')
    pairs = agree.pivot(index='contrast', columns='cell', values='n_pairs')

    fig, ax = plt.subplots(figsize=(1.15 * len(grid.columns) + 3,
                                    0.5 * len(grid) + 2.5))
    im = ax.imshow(grid.values, cmap='RdBu', vmin=0, vmax=1, aspect='auto')
    for i in range(len(grid)):
        for j in range(len(grid.columns)):
            ax.text(j, i, f'{int(labels.values[i, j])}/{int(pairs.values[i, j])}',
                    ha='center', va='center', fontsize=7)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, fontsize=7)
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.set_title(f'case{args.case}: pairs where the non-default level wins '
                 f'({args.metric})\nconcat_mean_last + mahalanobis is '
                 f'disqualified (findings/130 s2.4)')
    fig.colorbar(im, ax=ax, label='fraction of pairs positive')
    plt.tight_layout()
    fig.savefig(outdir / 'fig2_sign_agreement.png', dpi=150)
    plt.close(fig)


def fig_pareto(scores, args, outdir):
    """Figure 3. Accuracy against the two memory constraints."""
    sub = scores[scores.held_out_case == args.case].drop_duplicates(
        subset=['config_name', 'pooling', 'distance_head'])
    if 'total_flash_kb' not in sub:
        print('fig3: total_flash_kb missing, skipping')
        return

    panels = [('total_flash_kb', 'total flash: backbone int8 + head fp32 (KB)',
               RP2040_FLASH_KB),
              ('streaming_peak_ram_int8_kb', 'streaming peak RAM, int8 (KB)',
               RP2040_RAM_KB)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fronts = []

    for ax, (xcol, xlabel, limit) in zip(axes, panels):
        for head, g in sub.groupby('distance_head'):
            ax.scatter(g[xcol], g[args.metric], s=16, alpha=0.5, label=head)
        front = pareto_front(sub, xcol, args.metric)
        ax.plot(front[xcol], front[args.metric], color='black', linewidth=1.2,
                marker='o', markersize=4, zorder=3, label='frontier')
        ax.axvline(limit, color='#c0392b', linestyle='--', linewidth=1,
                   label=f'RP2040 ({limit} KB)')
        ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(args.metric)
        ax.legend(fontsize=7)
        fronts.append(front.assign(axis=xcol))

    pd.concat(fronts).to_csv(outdir / 'fig3_pareto_frontier.csv', index=False)
    fig.suptitle(f'case{args.case}: accuracy against memory')
    plt.tight_layout()
    fig.savefig(outdir / 'fig3_pareto.png', dpi=150)
    plt.close(fig)


def fig_thresholds(scores, args, outdir):
    """Figure 4. Which threshold methods place a usable threshold at all."""
    sub = scores[(scores.held_out_case == args.case)
                 & (scores.pooling == args.pooling)
                 & (scores.distance_head == args.head)].copy()
    if sub.empty:
        print('fig4: no rows, skipping')
        return
    sub['degenerate'] = sub.flags_nothing | sub.flags_everything
    sub.to_csv(outdir / 'fig4_threshold_metrics.csv', index=False)

    order = (sub.groupby('threshold_method')['degenerate'].mean()
                .sort_values().index.tolist())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 2.5))

    for i, method in enumerate(order):
        g = sub[sub.threshold_method == method]
        ok, bad = g[~g.degenerate], g[g.degenerate]
        for frame, colour in [(ok, '#2a78d6'), (bad, '#c0392b')]:
            if len(frame):
                ax.scatter(frame['f1'], i + rng.uniform(-0.12, 0.12, len(frame)),
                           s=18, alpha=0.6, color=colour)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([
        f"{m}  ({int(sub[sub.threshold_method == m]['degenerate'].sum())}"
        f"/{len(sub[sub.threshold_method == m])} degenerate)" for m in order],
        fontsize=8)
    ax.set_xlabel('F1, same-machine calibration')
    ax.set_title(f'case{args.case}  {args.pooling} + {args.head}\n'
                 f'red = flags nothing or flags everything')
    plt.tight_layout()
    fig.savefig(outdir / 'fig4_thresholds.png', dpi=150)
    plt.close(fig)


def fig_diagnostics(scores, diagnostics, args, outdir):
    """
    Figure 5. Whether the recurrence earns its keep.

    Half-life says how long the state could remember; the zero-state gap says
    whether that memory does anything. A config with a near-zero gap has a
    decorative scan and could be replaced by a convolution, which is a Phase 4
    finding worth more than a small AUC difference.
    """
    if diagnostics is None or diagnostics.empty:
        print('fig5: no diagnostics, skipping')
        return

    cell = one_cell(scores, args.case, args.pooling,
                    args.head)[['config_name', 'selective', args.metric]]
    # The collector writes the knobs into every diagnostic row, so a merge on
    # config_name alone would collide on `selective` and rename both sides to
    # _x and _y. Keep the knobs from `cell` and drop them here.
    d = (diagnostics[diagnostics.held_out_case == args.case]
         .drop(columns=[k for k in KNOBS if k in diagnostics], errors='ignore'))

    panels = [
        ('zero_state', 'state_contribution',
         'skill_real - skill_zeroed (one point per batch)', False),
        ('half_life', 'median_ms',
         'median half-life, ms (one point per layer)', True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dumps = []

    for ax, (check, metric, xlabel, logx) in zip(axes, panels):
        vals = d[(d.check == check) & (d.metric == metric)]
        merged = vals.merge(cell, on='config_name', how='inner')
        if merged.empty:
            ax.set_title(f'{check}: no rows')
            continue
        for sel, g in merged.groupby('selective'):
            ax.scatter(g['value'], g[args.metric], s=18, alpha=0.6,
                       label=f'selective={sel}')
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(args.metric)
        ax.legend(fontsize=8)
        dumps.append(merged.assign(panel=check))

    if dumps:
        pd.concat(dumps).to_csv(outdir / 'fig5_diagnostics.csv', index=False)
    fig.suptitle(f'case{args.case}  {args.pooling} + {args.head}')
    plt.tight_layout()
    fig.savefig(outdir / 'fig5_diagnostics.png', dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Phase 2 sweep figures')
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--pooling', default='mean')
    parser.add_argument('--head', default='knn_clustered_16')
    parser.add_argument('--metric', default='auc', choices=['auc', 'pauc'])
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()

    outdir = (PHASE2 / 'figures' if args.outdir is None
              else PROJECT_ROOT / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(PHASE2 / 'scores.csv')
    normalise_knobs(scores)
    diag_path = PHASE2 / 'diagnostics.csv'
    diagnostics = pd.read_csv(diag_path) if diag_path.exists() else None
    if diagnostics is not None:
        diagnostics = normalise_knobs(diagnostics)

    n_configs = scores[scores.held_out_case == args.case]['config_name'].nunique()
    print(f'case{args.case}: {n_configs} configs, {len(scores)} score rows')
    if n_configs < 72:
        print(f'WARNING: {72 - n_configs} configs missing. A hole unbalances '
              f'every contrast it belongs to; check sweep_logs.csv.')

    fig_paired_deltas(scores, args, outdir)
    fig_sign_agreement(scores, args, outdir)
    fig_pareto(scores, args, outdir)
    fig_thresholds(scores, args, outdir)
    fig_diagnostics(scores, diagnostics, args, outdir)
    print(f'\nwrote figures and exact tables to {outdir}')
    print(scores['selective'].unique())


if __name__ == '__main__':
    main()