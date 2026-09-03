"""
Signed effect of each knob step-down on the secondary metrics, across all
threshold-computation methods, for mean+euclidean and mean+knn_clustered_16.

Unlike AUC, the secondary metrics (accuracy, precision, recall, f1) depend on
where the threshold landed, so the knob effect is estimated separately per
threshold method: one OLS per (metric, head, threshold_method) over the 72
case-1 configs, with adjacent-step contrasts read via t_test. Output is one
heatmap per (metric, head): knob steps down the rows, threshold methods across
the columns, colour = signed change (positive = cheaper/simpler option scores
higher). Degenerate cells (a method that flags nothing or everything for most
configs) are masked, since a knob effect on a collapsed metric is meaningless.

Read f1 and accuracy as the real signals; precision and recall trade off as the
threshold moves, so a coloured recall row may reflect an operating-point shift
rather than a better model.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from runs.knob_cost import load_scores, eq, ROWS, KNOBS  # reuse the machinery

PHASE2 = PROJECT_ROOT / 'runs' / 'phase2'
METRICS = ['accuracy', 'precision', 'recall', 'f1']
CELLS = [('mean', 'euclidean'), ('mean', 'knn_clustered_16')]
# A method is unusable for a config if it flagged nothing or everything. If that
# holds for most configs in a cell, the whole (knob, method) effect is masked.
DEGENERATE_FRAC = 0.5


def fit_method_cell(scores, case, pooling, head, method, metric):
    sub = scores[eq(scores['held_out_case'], case)
                 & eq(scores['pooling'], pooling)
                 & eq(scores['distance_head'], head)
                 & eq(scores['threshold_method'], method)].copy()
    sub = sub.drop_duplicates(subset=['config_name'])
    if len(sub) < 10:
        return None, None
    # Mask if the method is degenerate for most configs in this cell.
    degen = (sub['flags_nothing'] | sub['flags_everything']).mean()
    if degen > DEGENERATE_FRAC:
        return None, None
    for col in ['d_state', 'expand', 'n_layers']:
        sub[col] = sub[col].astype(int)
    model = smf.ols(
        f'{metric} ~ C(selective) + C(d_state) + C(n_layers) '
        f'+ C(expand) + C(discretization)', data=sub).fit()
    return model, sub


def effect(model, contrast, sign):
    try:
        tt = model.t_test(contrast)
    except Exception:
        return np.nan
    return sign * float(np.ravel(tt.effect)[0])


def build_grid(scores, case, pooling, head, metric):
    methods = sorted(scores.loc[
        eq(scores['pooling'], pooling) & eq(scores['distance_head'], head),
        'threshold_method'].map(str).unique())
    labels = [r[0] for r in ROWS]
    grid = pd.DataFrame(index=labels, columns=methods, dtype=float)
    for method in methods:
        model, _ = fit_method_cell(scores, case, pooling, head, method, metric)
        if model is None:
            continue   # column stays NaN -> masked
        for label, contrast, sign, *_ in ROWS:
            grid.loc[label, method] = effect(model, contrast, sign)
    return grid


def plot_grid(grid, metric, pooling, head, outdir):
    fig, ax = plt.subplots(figsize=(0.62 * len(grid.columns) + 4,
                                    0.55 * len(grid) + 2.5))
    data = np.ma.masked_invalid(grid.values.astype(float))
    vmax = np.nanmax(np.abs(grid.values.astype(float))) or 0.1
    im = ax.imshow(data, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=6)

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.set_title(f'{metric}  |  {pooling} + {head}\n'
                 f'signed change per step (+ = cheaper/simpler better); '
                 f'blank = degenerate method', fontsize=10)
    fig.colorbar(im, ax=ax, label=f'change in {metric}')
    plt.tight_layout()
    out = outdir / f'secondary_{metric}_{pooling}_{head}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    grid.to_csv(outdir / f'secondary_{metric}_{pooling}_{head}.csv')


def main(case=1):
    scores = load_scores()
    outdir = PHASE2 / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        for pooling, head in CELLS:
            grid = build_grid(scores, case, pooling, head, metric)
            plot_grid(grid, metric, pooling, head, outdir)
            filled = grid.notna().sum().sum()
            print(f'{metric} {pooling}+{head}: {filled} cells filled')
    print(f'wrote 8 heatmaps and CSVs to {outdir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=int, default=1)
    main(p.parse_args().case)