"""
val_skill against AUC and pAUC, for every case-1 config, across all 12
pooling x head cells.

val_skill is the training objective (a next-frame prediction proxy); AUC/pAUC
are the detection metrics that actually matter. This plots whether the proxy
predicts the target. It does not, strongly: findings/130 and the case-1 sweep
show correlations in the 0.4-0.6 range, meaning val_skill cannot be used to
rank configs for deployment. This figure is the visual form of that claim.

val_skill is per-config (one training run, one value), so it comes from
configs.csv; AUC/pAUC are per (config, pooling, head), from scores.csv. Joined
on model_hash. Every point in one config's column shares an x-value.

Reads only the collected tables. Pearson r is annotated per panel; read it
against the seed band (SEED_SD, findings/210) -- a correlation is only
meaningful if the AUC spread it explains exceeds seed noise.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from runs.knob_cost import load_scores, eq, SEED_SD
from src.config import PROJECT_ROOT

PHASE2 = PROJECT_ROOT / 'runs' / 'phase2'
POOLINGS = ['mean', 'max', 'concat_mean_last']
HEADS = ['euclidean', 'mahalanobis', 'knn_full', 'knn_clustered_16']


def load_joined(case):
    """
    One row per (config, pooling, head) with val_skill attached.

    configs.csv may carry one row per (config, case) or per config; either way
    the join on model_hash brings val_skill onto every score row. .map(str) on
    model_hash guards the StringDtype-equality trap that has bitten the join
    keys before.
    """
    scores = load_scores()
    configs = pd.read_csv(PHASE2 / 'configs.csv')
    configs['model_hash'] = configs['model_hash'].map(str)
    scores['model_hash'] = scores['model_hash'].map(str)

    keep = configs.drop_duplicates('model_hash')[['model_hash', 'val_skill']]
    merged = scores[eq(scores['held_out_case'], case)].merge(
        keep, on='model_hash', how='left')
    merged = merged.drop_duplicates(subset=['config_name', 'pooling', 'distance_head'])
    return merged


def plot_metric(df, metric, case, outdir):
    fig, axes = plt.subplots(len(POOLINGS), len(HEADS),
                             figsize=(5 * len(HEADS), 4.2 * len(POOLINGS)),
                             squeeze=False)
    rows = []

    for i, pooling in enumerate(POOLINGS):
        for j, head in enumerate(HEADS):
            ax = axes[i, j]
            cell = df[eq(df['pooling'], pooling) & eq(df['distance_head'], head)]
            cell = cell.dropna(subset=['val_skill', metric])
            if len(cell) < 3:
                ax.set_title(f'{pooling} + {head}\n(n={len(cell)})', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            x = cell['val_skill'].values.astype(float)
            y = cell[metric].values.astype(float)
            r = np.corrcoef(x, y)[0, 1]
            rows.append({'pooling': pooling, 'head': head, 'metric': metric,
                         'pearson_r': r, 'n': len(cell)})

            ax.scatter(x, y, s=28, alpha=0.6, color='#2a78d6')
            # least-squares line, for the eye only; r is the reported number
            b, a = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, a + b * xs, color='#c0392b', lw=1, alpha=0.8)

            ax.set_title(f'{pooling} + {head}   r={r:+.2f}', fontsize=9)
            if i == len(POOLINGS) - 1:
                ax.set_xlabel('val_skill')
            if j == 0:
                ax.set_ylabel(metric)

    fig.suptitle(f'{metric} vs val_skill, case{case}, per pooling x head   |   '
                 f'r near zero = training proxy does not predict detection',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(outdir / f'skill_vs_{metric}_case{case}.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    return rows


def main(case=1):
    outdir = PHASE2 / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_joined(case)
    if df['val_skill'].isna().all():
        print('val_skill is all NaN after join; check configs.csv has the column '
              'and model_hash matches scores.csv')
        return
    print(f'{df["config_name"].nunique()} configs, {len(df)} (config,pooling,head) rows')

    summary = []
    for metric in ['auc', 'pauc']:
        summary.extend(plot_metric(df, metric, case, outdir))
    pd.DataFrame(summary).to_csv(outdir / f'skill_correlation_case{case}.csv',
                                 index=False)
    print(f'wrote skill_vs_auc_case{case}.png, skill_vs_pauc_case{case}.png, '
          f'and skill_correlation_case{case}.csv to {outdir}')
    print(f'\nseed band for reference: SEED_SD = {SEED_SD}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=int, default=1)
    main(p.parse_args().case)