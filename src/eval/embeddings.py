import json
import random

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

def get_embeddings(model, X, device, batch_size=128):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            emb = model(batch, mode='pooled')
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def get_embeddings_per_fold(held_out_case, cfg, device):
    dir_name = train_config_hash(cfg, held_out_case)
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base_dir / 'embeddings'
    out_dir.mkdir(parents=True, exist_ok=True)

    fold = get_fold(held_out_case)

    n_mels = cfg['features']['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels=n_mels)
    train_set = fold['train']
    val_normal_set = fold['val'][fold['val']['label'] == 'normal']
    test_set = fold['test']

    train_set['used_in'] = ['training' for _ in range(len(train_set))]
    val_normal_set['used_in'] = ['validation' for _ in range(len(val_normal_set))]
    test_set['used_in'] = ['test' for _ in range(len(test_set))]

    manifest = pd.concat([
        train_set,
        val_normal_set,
        test_set,
    ]).reset_index(drop=True)
    manifest.to_csv(str(out_dir / 'manifest.csv'), index=False)

    X_train = load_fold_clips(train_set, mean, std)
    X_val_normal = load_fold_clips(val_normal_set, mean, std)
    X_test = load_fold_clips(test_set, mean, std)
    test_labels = (fold['test']['label'].values == 'anomaly').astype(int)
    assert len(X_test) == len(test_labels), "clip/label count mismatch"

    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    for pooling_mode in POOLING_MODES:
        model.pooling_mode = pooling_mode

        train_emb = get_embeddings(model, X_train, device)
        val_normal_emb = get_embeddings(model, X_val_normal, device)
        test_emb = get_embeddings(model, X_test, device)

        np.savez(
            out_dir / f'emb_{pooling_mode}.npz',
            train_emb=train_emb,
            val_normal_emb=val_normal_emb,
            test_emb=test_emb,
            test_labels=test_labels,
            mean=mean,
            std=std,
        )

if __name__ == '__main__':
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cases = [1, 2, 3, 4]
    for case in cases:
        print(f'\n=== generating embeddings for case {case} ===')
        get_embeddings_per_fold(case, config, device)
