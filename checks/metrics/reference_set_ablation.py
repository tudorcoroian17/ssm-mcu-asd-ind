"""
Compares the reference population used by the distance heads: train-only
(current default), validation-normal-only, and train+val combined.

train_emb was directly optimized against during training, so its geometry
may not represent "normal" as honestly as embeddings the model never
received a gradient from. Validation normals come from the same three
non-held-out cases as train, so this is a reference-set design choice,
not a validity change -- neither option touches the held-out case.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from checks.metrics.eval_auc_pauc import (
    get_embeddings, score_euclidean, score_mahalanobis,
    score_knn_full, score_knn_clustered,
)

HEADS = {
    'euclidean': score_euclidean,
    'mahalanobis': score_mahalanobis,
    'knn_full': score_knn_full,
    'knn_clustered_16': score_knn_clustered,
}


def find_checkpoint(held_out_case):
    ckpts = sorted((PROJECT_ROOT / 'runs' / f'case{held_out_case}').glob('*.pt'))
    if len(ckpts) != 1:
        raise RuntimeError(f'expected 1 checkpoint in runs/case{held_out_case}, found {len(ckpts)}')
    return ckpts[0]


def describe_geometry(name, emb):
    centroid = emb.mean(axis=0)
    dists = np.linalg.norm(emb - centroid, axis=1)
    cov = np.cov(emb, rowvar=False)
    print(f"    {name:12s} n={len(emb):5d}  mean_dist_to_centroid={dists.mean():.4f}  "
          f"std_dist={dists.std():.4f}  cov_trace={np.trace(cov):.4f}  "
          f"cov_cond={np.linalg.cond(cov):.2e}")


def run(held_out_case, cfg, device, pooling='mean'):
    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels=n_mels)

    X_train = load_fold_clips(fold['train'], mean, std)
    val_normal_rows = fold['val'][fold['val']['label'] == 'normal']
    X_val_normal = load_fold_clips(val_normal_rows, mean, std)
    X_test = load_fold_clips(fold['test'], mean, std)
    labels = (fold['test']['label'].values == 'anomaly').astype(int)

    ckpt_path = find_checkpoint(held_out_case)
    model = SSMBackbone(**cfg['model']).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    model.eval()
    model.pooling = pooling

    train_emb = get_embeddings(model, X_train, device)
    val_emb = get_embeddings(model, X_val_normal, device)
    test_emb = get_embeddings(model, X_test, device)
    combined_emb = np.concatenate([train_emb, val_emb], axis=0)

    reference_sets = {'train': train_emb, 'val_normal': val_emb, 'train+val': combined_emb}

    print(f"\n{'=' * 92}")
    print(f"case{held_out_case} | pooling={pooling} | ckpt={ckpt_path.name}")
    print("  reference set geometry:")
    for ref_name, ref_emb in reference_sets.items():
        describe_geometry(ref_name, ref_emb)

    print("\n  scoring:")
    results = []
    for ref_name, ref_emb in reference_sets.items():
        for head_name, head_fn in HEADS.items():
            scores, extra = head_fn(ref_emb, test_emb, cfg['seed'])
            auc = roc_auc_score(labels, scores)
            pauc = roc_auc_score(labels, scores, max_fpr=0.1)
            cond_str = f"  cond={extra['cov_condition_number']:.2e}" if 'cov_condition_number' in extra else ''
            print(f"    {ref_name:12s} {head_name:20s} AUC={auc:.4f}  pAUC={pauc:.4f}{cond_str}")
            results.append({'ref': ref_name, 'head': head_name, 'auc': auc, 'pauc': pauc})
    return results


if __name__ == '__main__':
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for case in [1, 2, 3, 4]:
        run(case, cfg, device)