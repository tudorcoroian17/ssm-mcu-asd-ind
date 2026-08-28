from pathlib import Path

import numpy as np
import torch
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
    """1222010003_ToyCar_case2_ab22_IND_ch1_0003.wav -> 'ab22'."""
    return Path(path).stem.split('_')[3]


def find_checkpoint(held_out_case):
    ckpts = sorted((PROJECT_ROOT / 'runs' / f'case{held_out_case}').glob('*.pt'))
    if len(ckpts) != 1:
        raise RuntimeError(f'expected 1 checkpoint in runs/case{held_out_case}, found {len(ckpts)}: {ckpts}')
    return ckpts[0]


def analyze(held_out_case, cfg, device, pooling='mean'):
    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels=n_mels)

    X_train = load_fold_clips(fold['train'], mean, std)
    X_test = load_fold_clips(fold['test'], mean, std)
    labels = (fold['test']['label'].values == 'anomaly').astype(int)
    paths = fold['test']['path'].values

    ckpt_path = find_checkpoint(held_out_case)
    model = SSMBackbone(**cfg['model']).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    model.eval()
    model.pooling = pooling

    train_emb = get_embeddings(model, X_train, device)
    test_emb = get_embeddings(model, X_test, device)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    print(f"\n{'=' * 78}")
    print(f"case{held_out_case} | pooling={pooling} | ckpt={ckpt_path.name}")
    print(f"  {n_neg} normal, {n_pos} anomaly")

    buried_by_head = {}

    for head_name, head_fn in HEADS.items():
        scores, _ = head_fn(train_emb, test_emb, cfg['seed'])
        auc = roc_auc_score(labels, scores)

        normal_scores = scores[labels == 0]
        anomaly_idx = np.where(labels == 1)[0]

        # For each anomaly, how many normal clips outrank it. An anomaly
        # outranked by ALL normals is what the 1/265 arithmetic counts.
        n_above = np.array([(normal_scores > scores[i]).sum() for i in anomaly_idx])
        overlapping = anomaly_idx[n_above > 0]
        buried = anomaly_idx[n_above == n_neg]

        discordant = (1 - auc) * n_pos * n_neg
        print(f"\n  --- {head_name} | AUC={auc:.4f} ---")
        print(f"      discordant pairs = (1-AUC)*n_pos*n_neg = {discordant:.1f}")
        print(f"      anomalies below at least one normal: {len(overlapping)}")
        print(f"      anomalies below EVERY normal:        {len(buried)}")

        if len(overlapping):
            order = overlapping[np.argsort(-n_above[n_above > 0])]
            for i in order[:10]:
                n_out = (normal_scores > scores[i]).sum()
                flag = 'BURIED' if n_out == n_neg else f'{n_out}/{n_neg}'
                print(f"      {flag:>9s}  {fault_code(paths[i]):6s}  score={scores[i]:.4f}  "
                      f"{Path(paths[i]).name}")

        buried_by_head[head_name] = {fault_code(paths[i]) for i in buried}

    # Fault codes buried regardless of which head scores them are the
    # strongest candidates for a genuinely undetectable fault profile.
    common = set.intersection(*buried_by_head.values()) if buried_by_head else set()
    print(f"\n  buried under ALL four heads: {sorted(common) if common else '(none)'}")
    return buried_by_head


if __name__ == '__main__':
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_buried = {}
    for case in [1, 2, 3, 4]:
        all_buried[case] = analyze(case, cfg, device, pooling='mean')

    print(f"\n{'=' * 78}")
    print("fault codes buried under all four heads, per case:")
    for case, heads in all_buried.items():
        common = set.intersection(*heads.values()) if heads else set()
        print(f"  case{case}: {sorted(common) if common else '(none)'}")