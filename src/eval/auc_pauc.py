import json

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from runs.compute_hash import train_config_hash
from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone

POOLING_MODES = ['mean', 'max', 'concat_mean_last']

def read_embeddings(held_out_case, config, pooling_mode):
    dir_name = train_config_hash(config, held_out_case)
    embeddings_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name / 'embeddings'

    embeddings = np.load(str(embeddings_dir / f'emb_{pooling_mode}.npz'))
    return (embeddings['train_emb'], embeddings['val_normal_emb'], embeddings['test_emb'],
            embeddings['test_labels'], embeddings['mean'], embeddings['std'])


def score_euclidean(train_emb, test_emb, seed):
    centroid = train_emb.mean(axis=0)
    return np.linalg.norm(test_emb - centroid, axis=1), {'centroid': centroid}

def score_mahalanobis(train_emb, test_emb, seed):
    centroid = train_emb.mean(axis=0)
    cov = np.cov(train_emb, rowvar=False)

    cond = np.linalg.cond(cov)
    if cond > 1e6:
        print(f"    WARNING: covariance condition number {cond:.2e} -- Mahalanobis scores may be unstable")

    inv_cov = np.linalg.inv(cov)
    diff = test_emb - centroid
    scores = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
    return scores, {'centroid': centroid, 'cov_condition_number': float(cond)}

def score_knn_full(train_emb, test_emb, seed, k=5):
    dists = cdist(test_emb, train_emb)
    return np.sort(dists, axis=1)[:, k - 1], {}


def score_knn_clustered(train_emb, test_emb, seed, n_references=16, k=1):
    kmeans = KMeans(n_clusters=n_references, n_init=10, random_state=seed).fit(train_emb)
    references = kmeans.cluster_centers_
    dists = cdist(test_emb, references)
    return np.sort(dists, axis=1)[:, min(k, n_references) - 1], {}

DISTANCE_HEADS = {
    'euclidean': score_euclidean,
    'mahalanobis': score_mahalanobis,
    'knn_full': score_knn_full,
    'knn_clustered_16': score_knn_clustered,
}

def plot_score_distribution(scores, test_labels, title, out_path):
    normal_scores = scores[test_labels == 0]
    anomaly_scores = scores[test_labels == 1]
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=30, alpha=0.6, label='normal', color='#2a78d6')
    plt.hist(anomaly_scores, bins=30, alpha=0.6, label='anomaly', color='#d62a2a')
    plt.xlabel('anomaly score')
    plt.ylabel('count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(scores, test_labels, auc, title, out_path):
    fpr, tpr, _ = roc_curve(test_labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='#2a78d6', label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='chance')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_embedding_pca(train_emb, test_emb, test_labels, centroid, title, out_path):
    pca = PCA(n_components=2).fit(train_emb)
    test_2d = pca.transform(test_emb)
    centroid_2d = pca.transform(centroid.reshape(1, -1))

    plt.figure(figsize=(7, 6))
    plt.scatter(test_2d[test_labels == 0, 0], test_2d[test_labels == 0, 1],
                alpha=0.5, label='normal', color='#2a78d6', s=15)
    plt.scatter(test_2d[test_labels == 1, 0], test_2d[test_labels == 1, 1],
                alpha=0.5, label='anomaly', color='#d62a2a', s=15)
    plt.scatter(*centroid_2d[0], marker='*', s=300, color='black', label='centroid', zorder=5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_case_eval(held_out_case, cfg, device):
    dir_name = train_config_hash(cfg, held_out_case)
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name
    eval_dir = base_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = base_dir / 'scores'
    scores_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for pooling_mode in POOLING_MODES:
        print(f"\n=== case{held_out_case} | pooling: {pooling_mode} ===")
        train_emb, _, test_emb, test_labels, _, _ = read_embeddings(held_out_case, cfg, pooling_mode)

        centroid = train_emb.mean(axis=0)
        plot_embedding_pca(
            train_emb, test_emb, test_labels, centroid,
            title=f'case{held_out_case} [{pooling_mode}]: embedding space (PCA)',
            out_path=eval_dir / f'embedding_pca_{pooling_mode}.png',
        )

        for head_name, head_fn in DISTANCE_HEADS.items():
            scores, extra = head_fn(train_emb, test_emb, cfg['seed'])

            np.savez(
                scores_dir / f'scores_{pooling_mode}_{head_name}.npz',
                scores=scores,
            )

            auc = roc_auc_score(test_labels, scores)
            pauc = roc_auc_score(test_labels, scores, max_fpr=0.1)
            print(f"  {head_name:20s} AUC={auc:.4f}  pAUC={pauc:.4f}")

            suffix = f"{pooling_mode}_{head_name}"
            plot_score_distribution(
                scores, test_labels,
                title=f'case{held_out_case} [{pooling_mode}/{head_name}]: score distribution (AUC={auc:.3f})',
                out_path=eval_dir / f'score_distribution_{suffix}.png',
            )
            plot_roc(
                scores, test_labels, auc,
                title=f'case{held_out_case} [{pooling_mode}/{head_name}]: ROC',
                out_path=eval_dir / f'roc_curve_{suffix}.png',
            )

            results.append({
                'held_out_case': held_out_case,
                'pooling': pooling_mode,
                'distance_head': head_name,
                'auc': float(auc),
                'pauc': float(pauc),
                **{k: v for k, v in extra.items() if k == 'cov_condition_number'},
            })

    with open(eval_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    data = [(r['held_out_case'], r['pooling'], r['distance_head'], r['auc'], r['pauc']) for r in results]
    df = pd.DataFrame(data, columns=['held_out_case', 'pooling', 'distance_head', 'auc', 'pauc'])
    df.to_csv(eval_dir / 'results.csv', index=False)

    return results

def print_summary(res):
    print("\n=== summary ===")
    for r in sorted(res, key=lambda r: -r['auc']):
        print(f"{r['pooling']:20s} {r['distance_head']:20s} AUC={r['auc']:.4f}  pAUC={r['pauc']:.4f}")


if __name__ == "__main__":
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cases = [1, 2, 3, 4]
    for case in cases:
        print(f'\n=== held_out_case {case} ===')
        results = run_case_eval(case, cfg, device)
        print_summary(results)