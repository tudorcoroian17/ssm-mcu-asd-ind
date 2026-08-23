import json
import numpy as np

from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.data.folds import get_fold, CASE_IDS
from src.features.stats import compute_normalization_stats

FOLD_STATS_PATH = PROJECT_ROOT / 'configs' / 'all_folds_normalisation_stats.json'
_cfg = load_config()
_training = _cfg['training']
_features = _cfg['features']
horizon_k = _training['horizon_k']

with open(FOLD_STATS_PATH) as f:
    all_folds_normalization_stats = json.load(f)

def apply_normalization(x, mean, std):
    return (x - mean) / std

def load_fold_clips(rows, mean, std):
    # Stack every clip in this fold split into one (n_clips, T, n_mels)
    arrays = [np.load(p) for p in rows['cache_path']]
    T_values = {a.shape[0] for a in arrays}
    assert len(T_values) == 1, f"expected one T across all clips, got {T_values}"
    normalize = [apply_normalization(a, mean, std) for a in arrays]
    return np.stack(normalize)

def compute_baselines(x_train, k):
    assert x_train.ndim == 3, f'expected (n_clips, T, n_mels), got {x_train.shape}'
    n_clips, T, n_mels = x_train.shape
    assert T > k, f'horizon k = {k} must be < sequence length T = {T}'

    future = x_train[:, k:]
    current = x_train[:, :-k]
    mse_persistence = float(((future - current) ** 2).mean())

    mu = x_train.mean(axis=(0, 1))
    mse_climatology = float (((future - mu) ** 2).mean())

    return {
        'n_clips': int(n_clips),
        'T': int(T),
        'n_mels': int(n_mels),
        'horizon_k': int(k),
        'mse_persistence': mse_persistence,
        'mse_climatology': mse_climatology,
        'mu_frame_norm': float(np.linalg.norm(mu)),
        'ratio': mse_persistence / mse_climatology,
    }

def compute_fold_baselines(held_out_case, k):
    fold = get_fold(held_out_case)
    n_mels = _features['n_mels']
    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels)
    x_train = load_fold_clips(fold['train'], mean, std)
    result = compute_baselines(x_train, k)
    result['held_out_case'] = held_out_case
    return result

if __name__ == '__main__':
    out_dir = PROJECT_ROOT / 'runs' / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for case in CASE_IDS:
        print(f'--- held-out case: {case} ---')
        result = compute_fold_baselines(case, horizon_k)
        print(result)
        all_results[f'case{case}'] = result

    with open(out_dir / f'toycar_all_folds_k{horizon_k}_baselines.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    ratios = [r['ratio'] for r in all_results.values()]
    print(f"\nratio: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")


