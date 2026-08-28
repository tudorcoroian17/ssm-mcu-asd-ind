"""
Test whether fusing Euclidean and Mahalanobis scores recovers the residual
gap left by each head alone.

Motivation: the two heads fail on disjoint fault families. Euclidean misses
steel-ribbon faults (ab06, ab07); Mahalanobis misses melted-gear + plastic
+ under-voltage faults (ab22, ab49). Neither head's failures survive the
other, so a combined score should outperform both.

Two fusion variants:
  rank  -- transductive. Ranks test scores within the test set. Cannot run
           per-clip on an MCU; included as an upper bound.
  zscore -- calibrated on the fold's validation normals (never the held-out
           case). Two scalars per head, fixed at training time. Deployable.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from src.eval.auc_pauc import (
    get_embeddings, score_euclidean, score_mahalanobis,
    score_knn_full, score_knn_clustered,
)

HEADS = {
    'euclidean': score_euclidean,
    'mahalanobis': score_mahalanobis,
    'knn_full': score_knn_full,
    'knn_clustered_16': score_knn_clustered,
}


def fault_code(path):
    return Path(path).stem.split('_')[3]


def find_checkpoint(held_out_case):
    ckpts = sorted((PROJECT_ROOT / 'runs' / f'case{held_out_case}').glob('*.pt'))
    if len(ckpts) != 1:
        raise RuntimeError(f'expected 1 checkpoint in runs/case{held_out_case}, found {len(ckpts)}')
    return ckpts[0]


def buried_summary(scores, labels, paths):
    """Anomalies ranked below every normal clip, with their fault codes."""
    normal_scores = scores[labels == 0]
    anomaly_idx = np.where(labels == 1)[0]
    buried = [i for i in anomaly_idx if (normal_scores > scores[i]).sum() == len(normal_scores)]
    return len(buried), sorted({fault_code(paths[i]) for i in buried})


def evaluate(name, scores, labels, paths, indent='  '):
    auc = roc_auc_score(labels, scores)
    pauc = roc_auc_score(labels, scores, max_fpr=0.1)
    n_buried, codes = buried_summary(scores, labels, paths)
    codes_str = ','.join(codes) if codes else '-'
    print(f"{indent}{name:34s} AUC={auc:.4f}  pAUC={pauc:.4f}  buried={n_buried:2d}  {codes_str}")
    return {'name': name, 'auc': auc, 'pauc': pauc, 'n_buried': n_buried}


def run_case(held_out_case, cfg, device, pooling='mean'):
    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels=n_mels)

    X_train = load_fold_clips(fold['train'], mean, std)
    X_test = load_fold_clips(fold['test'], mean, std)
    labels = (fold['test']['label'].values == 'anomaly').astype(int)
    paths = fold['test']['path'].values

    # Calibration set: validation normals from the TRAINING cases only.
    # Never touches the held-out case, so z-scoring stays honest.
    val_normal_rows = fold['val'][fold['val']['label'] == 'normal']
    X_calib = load_fold_clips(val_normal_rows, mean, std)

    ckpt_path = find_checkpoint(held_out_case)
    model = SSMBackbone(**cfg['model']).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    model.eval()
    model.pooling = pooling

    train_emb = get_embeddings(model, X_train, device)
    test_emb = get_embeddings(model, X_test, device)
    calib_emb = get_embeddings(model, X_calib, device)

    print(f"\n{'=' * 92}")
    print(f"case{held_out_case} | pooling={pooling} | ckpt={ckpt_path.name} | "
          f"{int((labels == 0).sum())} normal, {int(labels.sum())} anomaly, "
          f"{len(X_calib)} calibration clips")

    raw, calib_stats = {}, {}
    print("\n  individual heads:")
    for head_name, head_fn in HEADS.items():
        scores, _ = head_fn(train_emb, test_emb, cfg['seed'])
        calib_scores, _ = head_fn(train_emb, calib_emb, cfg['seed'])
        raw[head_name] = scores
        calib_stats[head_name] = (float(calib_scores.mean()), float(calib_scores.std()))
        evaluate(head_name, scores, labels, paths)

    results = []
    for a, b in combinations(HEADS, 2):
        print(f"\n  {a} + {b}:")

        # Transductive upper bound. rankdata over the test set only.
        rank_fused = (rankdata(raw[a]) + rankdata(raw[b])) / (2 * len(labels))
        results.append(evaluate(f'rank_mean({a},{b})', rank_fused, labels, paths))

        # Deployable. Calibration constants come from validation normals.
        za = (raw[a] - calib_stats[a][0]) / calib_stats[a][1]
        zb = (raw[b] - calib_stats[b][0]) / calib_stats[b][1]
        results.append(evaluate(f'z_mean({a},{b})', (za + zb) / 2, labels, paths))
        results.append(evaluate(f'z_max({a},{b})', np.maximum(za, zb), labels, paths))

    return results


if __name__ == '__main__':
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for case in [1, 2, 3, 4]:
        all_results[case] = run_case(case, cfg, device, pooling='mean')

    print(f"\n{'=' * 92}")
    print("best fusion per case:")
    for case, results in all_results.items():
        best = max(results, key=lambda r: r['auc'])
        print(f"  case{case}: {best['name']:34s} AUC={best['auc']:.4f}  "
              f"pAUC={best['pauc']:.4f}  buried={best['n_buried']}")