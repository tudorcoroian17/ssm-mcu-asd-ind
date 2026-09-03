"""
Ranks the Phase 2 training knobs by how much AUC each single step-down costs,
and by that cost against the flash the step frees.

Every knob here reduces model footprint, so the deployment question is not
"which knob helps" but "which step can I take most freely". Each three-level
knob is split into adjacent steps (d_state 32->16 and 16->8; n_layers 4->2 and
2->1) rather than measured against the cheapest level, so a row is one real
design decision. Two panels per (pooling, head):

  top     signed change per step, sorted, wandering around zero. A bar right of
          zero means the cheaper/simpler option scored higher.
  bottom  that change against flash freed: a step high and far right costs the
          most accuracy for the most memory saved (expensive to remove); a step
          near zero and far right is a near-free saving.

Coefficients come from one OLS per (pooling, head)
(auc ~ C(selective)+C(d_state)+C(n_layers)+C(expand)+C(discretization)) over the
72 case-1 configs. Adjacent-step contrasts are evaluated with model.t_test, so a
step like 32->16 gets its own standard error rather than one pieced together
from two against-baseline coefficients. Signs are flipped so positive = the
cheaper/simpler option scores higher. The grey band is the case-1 seed spread
from findings/210 (SEED_SD); a step inside it costs nothing distinguishable from
seed noise.

selective has no additive main effect, only an interaction (case-1 paired deltas
span -0.32 to +0.21). Its bar is the average over that interaction and sits near
zero. That is the right reading for "cheapest to turn on average" but hides that
the cost is large in some corners; keep the interaction figure alongside.

Note on dtypes: pandas types text columns as StringDtype, and `column == a_str`
on that dtype returns pd.NA (falsy) rather than False, which silently empties
every filter and pivot. Every string column is forced to plain object at read
time, and all filtering goes through eq(), which coerces both sides to str.
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

PHASE2 = PROJECT_ROOT / 'runs' / 'phase2'
SEED_SD = 0.0132   # case-1 seed spread, findings/210
KNOBS = ['d_state', 'n_layers', 'expand', 'selective', 'discretization']
STRING_COLS = ['pooling', 'distance_head', 'discretization', 'config_name',
               'threshold_method']
POOLINGS = ['mean', 'max', 'concat_mean_last']
HEADS = ['euclidean', 'mahalanobis', 'knn_full', 'knn_clustered_16']

# (label, t_test contrast expression, sign so positive = cheaper/simpler better,
#  knob, cheap level of the step, expensive level of the step)
# Each contrast is one adjacent step. Two-level knobs have a single step.
ROWS = [
    ('selective: mamba->classic',  'C(selective)[T.True]',                   -1, 'selective',      False,   True),
    ('d_state: 32->16',            'C(d_state)[T.32] - C(d_state)[T.16]',    -1, 'd_state',        16,      32),
    ('d_state: 16->8',             'C(d_state)[T.16]',                       -1, 'd_state',        8,       16),
    ('n_layers: 4->2',             'C(n_layers)[T.4] - C(n_layers)[T.2]',    -1, 'n_layers',       2,       4),
    ('n_layers: 2->1',             'C(n_layers)[T.2]',                       -1, 'n_layers',       1,       2),
    ('expand: 2->1',               'C(expand)[T.2]',                         -1, 'expand',         1,       2),
    ('discretization: zoh->euler', 'C(discretization)[T.zoh]',              -1, 'discretization', 'euler', 'zoh'),
]

# One fixed colour per step, shared across all figures so a step reads the same
# in every panel. Within-knob pairs share a hue family (two blues, two greens)
# so they still group by eye, split light/dark to tell the two steps apart.
STEP_COLOURS = {
    'selective: mamba->classic':  '#c0392b',   # red
    'd_state: 32->16':            '#1f5fa8',   # dark blue
    'd_state: 16->8':             '#5dade2',   # light blue
    'n_layers: 4->2':             '#1e8449',   # dark green
    'n_layers: 2->1':             '#58d68d',   # light green
    'expand: 2->1':               '#e67e22',   # orange
    'discretization: zoh->euler': '#8e44ad',   # purple
}


def load_scores():
    """Read scores.csv with every string column forced to plain Python str."""
    s = pd.read_csv(PHASE2 / 'scores.csv')
    for col in STRING_COLS:
        if col in s:
            s[col] = s[col].map(str)
    return s


def eq(series, value):
    """
    Equality that survives pandas' StringDtype.

    series.map(str) forces plain strings, so the comparison returns real
    booleans instead of pd.NA. Use this everywhere instead of `series == value`.
    """
    return series.map(str) == str(value)


def fit_cell(scores, case, pooling, head, metric):
    sub = scores[eq(scores['held_out_case'], case)
                 & eq(scores['pooling'], pooling)
                 & eq(scores['distance_head'], head)].copy()
    sub = sub.drop_duplicates(subset=['config_name'])
    if len(sub) < 10:
        return None, None
    sub['d_state'] = sub['d_state'].astype(int)
    sub['expand'] = sub['expand'].astype(int)
    sub['n_layers'] = sub['n_layers'].astype(int)
    model = smf.ols(
        f'{metric} ~ C(selective) + C(d_state) + C(n_layers) '
        f'+ C(expand) + C(discretization)',
        data=sub).fit()
    return model, sub


def flash_step(sub, knob, cheap_level, expensive_level):
    """
    Mean backbone flash freed by one adjacent step down in `knob`, from exact
    one-flip pairs holding the other four knobs fixed. Reads
    flash_footprint_int8_kb; the distance head is excluded because it does not
    vary with these architecture knobs.
    """
    others = [k for k in KNOBS if k != knob]
    sub = sub.copy()
    sub[knob] = sub[knob].map(str)
    wide = sub.pivot(index=others, columns=knob,
                     values='flash_footprint_int8_kb')
    lo, hi = str(cheap_level), str(expensive_level)
    if lo not in wide.columns or hi not in wide.columns:
        return np.nan
    paired = wide.loc[:, [lo, hi]].dropna()
    if paired.empty:
        return np.nan
    return float((paired[hi] - paired[lo]).mean())


def cell_rows(model, sub, metric, pooling, head):
    out = []
    for label, contrast, sign, knob, cheap, expensive in ROWS:
        # t_test evaluates the linear combination with its own SE, so an
        # adjacent-step contrast like 32->16 gets a correct CI rather than one
        # pieced together from two against-baseline coefficients.
        try:
            tt = model.t_test(contrast)
        except Exception:
            continue
        est = float(np.ravel(tt.effect)[0])
        lo_ci, hi_ci = np.ravel(tt.conf_int(alpha=0.05))[:2]
        lo, hi = sorted([sign * lo_ci, sign * hi_ci])
        out.append({
            'pooling': str(pooling), 'head': str(head), 'metric': metric,
            'row': label,
            'coef': sign * est, 'lo': lo, 'hi': hi,
            'abs_coef': abs(est),
            'flash_saved_kb': flash_step(sub, knob, cheap, expensive),
            'p': float(np.ravel(tt.pvalue)[0]),
        })
    return out


def build_coefficients(scores, case, metric):
    rows = []
    for pooling in POOLINGS:
        for head in HEADS:
            model, sub = fit_cell(scores, case, pooling, head, metric)
            if model is not None:
                rows.extend(cell_rows(model, sub, metric, pooling, head))
    return pd.DataFrame(rows)


def plot_pooling(coefs, pooling, metric, outdir):
    cells = [(head, coefs[eq(coefs['pooling'], pooling) & eq(coefs['head'], head)])
             for head in HEADS]
    if all(cell.empty for _, cell in cells):
        print(f'{pooling}: no rows, skipping')
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), squeeze=False)

    for j, (head, cell) in enumerate(cells):
        top, bot = axes[0, j], axes[1, j]
        if cell.empty:
            top.set_title(f'{head}\n(no rows)', fontsize=10)
            continue

        # top: signed change per step, sorted, wandering around zero
        c = cell.sort_values('coef').reset_index(drop=True)
        top.axvspan(-SEED_SD, SEED_SD, color='grey', alpha=0.15)
        top.axvline(0, color='black', linewidth=0.8)
        bar_colours = [STEP_COLOURS[lbl] for lbl in c['row'].values]
        top.barh(range(len(c)), c['coef'].values, color=bar_colours)
        top.errorbar(c['coef'].values, range(len(c)),
                     xerr=[(c['coef'] - c['lo']).values,
                           (c['hi'] - c['coef']).values],
                     fmt='none', ecolor='black', elinewidth=0.7, capsize=2)
        top.set_yticks(range(len(c)))
        top.set_yticklabels(c['row'].values if j == 0 else [], fontsize=8)
        top.set_xlabel(f'change in {metric}\n(+ = cheaper/simpler better)')
        top.set_title(head, fontsize=11)
        top.set_ylim(-0.6, len(c) - 0.4)
        top.invert_yaxis()

        # bottom: signed cost against flash freed, coloured per step
        bot.axhspan(-SEED_SD, SEED_SD, color='grey', alpha=0.15)
        bot.axhline(0, color='black', linewidth=0.7)
        x = np.nan_to_num(cell['flash_saved_kb'].values.astype(float))
        y = cell['coef'].values.astype(float)
        colours = [STEP_COLOURS[lbl] for lbl in cell['row'].values]
        bot.scatter(x, y, s=55, c=colours, zorder=3, edgecolors='white',
                    linewidths=0.5)
        for xi, yi, lbl in zip(x, y, cell['row'].values):
            step = lbl.split(':')[1].strip()
            bot.annotate(step, (xi, yi), fontsize=6.5,
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
    fig.suptitle(f'pooling = {pooling}   |   grey = seed noise '
                 f'(+/-{SEED_SD}, findings/210)   |   selective is an '
                 f'interaction; its bar is the average', fontsize=12)
    fig.savefig(outdir / f'knob_cost_{metric}_{pooling}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(metric='auc', case=1):
    scores = load_scores()
    outdir = PHASE2 / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)

    coefs = build_coefficients(scores, case, metric)
    print(f'built {len(coefs)} coefficient rows')
    if coefs.empty:
        print('no coefficients; check that scores.csv has case', case)
        return

    coefs.to_csv(outdir / f'knob_cost_{metric}_coefficients.csv', index=False)
    for pooling in POOLINGS:
        plot_pooling(coefs, pooling, metric, outdir)
    print(f'wrote 3 PNGs and knob_cost_{metric}_coefficients.csv to {outdir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metric', default='auc', choices=['auc', 'pauc'])
    p.add_argument('--case', type=int, default=1)
    a = p.parse_args()
    main(a.metric, a.case)