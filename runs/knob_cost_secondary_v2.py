"""
Signed effect of each knob step-down on a secondary metric, pooled across
threshold-computation methods, with flash paired in. For mean+euclidean and
mean+knn_clustered_16.

The secondary metrics (accuracy, precision, recall, f1) depend on where the
threshold landed. This figure pools the threshold methods: threshold_method
enters the OLS as a factor, so each knob coefficient is its effect on the metric
averaged over operating points, holding method fixed. Degenerate rows (a method
that flagged nothing or everything for a config) are dropped before fitting,
because a metric that is structurally collapsed carries no knob signal and only
dilutes the pooled estimate toward zero.

Layout per metric: two columns (euclidean, knn_clustered_16), two rows.
  top     signed change per step, sorted, wandering around zero. Right of zero =
          the cheaper/simpler option scored higher.
  bottom  that change against backbone flash freed (int8). The flash axis is the
          same backbone number as the AUC figure; it does not vary with the
          metric or the threshold method, and is repeated here only for the
          cost-versus-saving read.

Adjacent-step contrasts (d_state 32->16, 16->8; n_layers 4->2, 2->1) are read via
t_test for correct standard errors. Read f1 and accuracy as the real signals;
precision and recall trade off as the threshold moves, so a coloured recall bar
may reflect an operating-point shift rather than a better model.
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import PROJECT_ROOT
from runs.knob_cost import (load_scores, eq, flash_step, ROWS, STEP_COLOURS,
                            SEED_SD)

PHASE2 = PROJECT_ROOT / 'runs' / 'phase2'
METRICS = ['accuracy', 'precision', 'recall', 'f1']
HEADS = ['euclidean', 'knn_clustered_16']
POOLING = 'mean'


def fit_pooled(scores, case, head, metric):
    """
    One OLS over all threshold methods for a (pooling, head) cell, with
    threshold_method as a factor. Degenerate rows dropped first.
    """
    sub = scores[eq(scores['held_out_case'], case)
                 & eq(scores['pooling'], POOLING)
                 & eq(scores['distance_head'], head)].copy()
    sub = sub[~(sub['flags_nothing'] | sub['flags_everything'])]
    if len(sub) < 20:
        return None, None
    for col in ['d_state', 'expand', 'n_layers']:
        sub[col] = sub[col].astype(int)
    model = smf.ols(
        f'{metric} ~ C(selective) + C(d_state) + C(n_layers) + C(expand) '
        f'+ C(discretization) + C(threshold_method)', data=sub).fit()
    return model, sub


def cell_rows(model, sub, metric, head):
    out = []
    for label, contrast, sign, knob, cheap, expensive in ROWS:
        try:
            tt = model.t_test(contrast)
        except Exception:
            continue
        est = float(np.ravel(tt.effect)[0])
        lo_ci, hi_ci = np.ravel(tt.conf_int(alpha=0.05))[:2]
        lo, hi = sorted([sign * lo_ci, sign * hi_ci])
        out.append({
            'head': head, 'metric': metric, 'row': label,
            'coef': sign * est, 'lo': lo, 'hi': hi,
            # flash uses the full sub, deduped to one row per config, so the
            # pairing is over configs not over the repeated method rows.
            'flash_saved_kb': flash_step(
                sub.drop_duplicates(subset=['config_name']),
                knob, cheap, expensive),
        })
    return out


def plot_metric(coefs, metric, outdir):
    fig, axes = plt.subplots(2, len(HEADS), figsize=(9 * len(HEADS), 10),
                             squeeze=False)
    for j, head in enumerate(HEADS):
        cell = coefs[eq(coefs['head'], head) & eq(coefs['metric'], metric)]
        top, bot = axes[0, j], axes[1, j]
        if cell.empty:
            top.set_title(f'{head}\n(no rows)', fontsize=10)
            continue

        c = cell.sort_values('coef').reset_index(drop=True)
        top.axvspan(-SEED_SD, SEED_SD, color='grey', alpha=0.15)
        top.axvline(0, color='black', linewidth=0.8)
        top.barh(range(len(c)), c['coef'].values,
                 color=[STEP_COLOURS[l] for l in c['row'].values])
        top.errorbar(c['coef'].values, range(len(c)),
                     xerr=[(c['coef'] - c['lo']).values,
                           (c['hi'] - c['coef']).values],
                     fmt='none', ecolor='black', elinewidth=0.7, capsize=2)
        top.set_yticks(range(len(c)))
        top.set_yticklabels(c['row'].values if j == 0 else [], fontsize=8)
        top.set_xlabel(f'change in {metric}\n(+ = cheaper/simpler better)')
        top.set_title(f'{POOLING} + {head}', fontsize=11)
        top.set_ylim(-0.6, len(c) - 0.4)
        top.invert_yaxis()

        x = np.nan_to_num(cell['flash_saved_kb'].values.astype(float))
        y = cell['coef'].values.astype(float)
        bot.axhspan(-SEED_SD, SEED_SD, color='grey', alpha=0.15)
        bot.axhline(0, color='black', linewidth=0.7)
        bot.scatter(x, y, s=55, c=[STEP_COLOURS[l] for l in cell['row'].values],
                    zorder=3, edgecolors='white', linewidths=0.5)
        for xi, yi, lbl in zip(x, y, cell['row'].values):
            bot.annotate(lbl.split(':')[1].strip(), (xi, yi), fontsize=6.5,
                         xytext=(4, 3), textcoords='offset points')
        xpad = max(np.abs(x).max() * 0.15, 1.0)
        ypad = max(np.abs(y).max() * 0.3, SEED_SD * 1.5)
        bot.set_xlim(x.min() - xpad, x.max() + xpad)
        bot.set_ylim(y.min() - ypad, y.max() + ypad)
        bot.set_xlabel('backbone flash freed (KB, int8)')
        if j == 0:
            bot.set_ylabel(f'change in {metric}\n(+ = cheaper/simpler better)')

    handles = [Patch(color=col, label=lbl) for lbl, col in STEP_COLOURS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f'{metric}, pooled over threshold methods (degenerate dropped) '
                 f'|  grey = seed noise (+/-{SEED_SD}, findings/210)', fontsize=12)
    fig.savefig(outdir / f'secondary_{metric}_pooled.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def main(case=1):
    scores = load_scores()
    outdir = PHASE2 / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for metric in METRICS:
        for head in HEADS:
            model, sub = fit_pooled(scores, case, head, metric)
            if model is not None:
                all_rows.extend(cell_rows(model, sub, metric, head))
    coefs = pd.DataFrame(all_rows)
    coefs.to_csv(outdir / 'secondary_pooled_coefficients.csv', index=False)
    print(f'built {len(coefs)} rows (expect {len(METRICS)*len(HEADS)*7} = '
          f'{len(METRICS)*len(HEADS)*7})')
    for metric in METRICS:
        plot_metric(coefs, metric, outdir)
    print(f'wrote {len(METRICS)} PNGs and secondary_pooled_coefficients.csv')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=int, default=1)
    main(p.parse_args().case)