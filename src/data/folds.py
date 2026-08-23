import pandas as pd
import numpy as np
import json
from src.config import PROJECT_ROOT, load_config
from src.features.stats import compute_normalization_stats

MANIFEST_PATH = PROJECT_ROOT / 'manifest.csv'
CASE_IDS = [1, 2, 3, 4]
_cfg = load_config()

def get_fold(held_out_case):
    assert held_out_case in CASE_IDS

    train_cases = [c for c in CASE_IDS if c != held_out_case]
    manifest = pd.read_csv(MANIFEST_PATH)

    _data = _cfg['data']
    eval_balance_ratio = _data['eval_balance_ratio']
    eval_balance_seed = _data['eval_balance_seed']
    val_fraction = _data['val_fraction']
    val_split_seed = _data['val_split_seed']

    # only for training and validation
    ind_normal_pool = manifest[
        manifest['case_id'].isin(train_cases)
        & (manifest['source'] == 'IND') & (manifest['label'] == 'normal')
    ]
    # only for computing thresholds for metrics
    ind_anomaly_pool = manifest[
        manifest['case_id'].isin(train_cases)
        & (manifest['source'] == 'IND') & (manifest['label'] == 'anomaly')
    ]

    rng = np.random.default_rng(val_split_seed)

    shuffled = ind_normal_pool.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    n_val = int(round(len(shuffled) * val_fraction))
    val_normal_rows = shuffled.iloc[:n_val]
    train_rows = shuffled.iloc[n_val:]

    # only for testing
    test_normal = manifest[(manifest['case_id'] == held_out_case) &
                           (manifest['source'] == 'IND') & (manifest['label'] == 'normal')]
    test_anomaly = manifest[(manifest['case_id'] == held_out_case) &
                            (manifest['source'] == 'IND') & (manifest['label'] == 'anomaly')]
    test_rng = np.random.default_rng(eval_balance_seed)
    n_normal = int(round(len(test_anomaly) * eval_balance_ratio))
    test_normal_balanced = test_normal.sample(n=n_normal, random_state=test_rng)
    test_rows = pd.concat([test_normal_balanced, test_anomaly]).reset_index(drop=True)

    return {
        'train': train_rows,
        'val': pd.concat([val_normal_rows, ind_anomaly_pool]).reset_index(drop=True),
        'test': test_rows,
        'val_fraction': val_fraction, 'split_seed': val_split_seed,
        'balance_seed': eval_balance_seed, 'balance_ratio': eval_balance_ratio,
    }

def build_fold_stats():
    _features = _cfg['features']
    n_mels = _features['n_mels']
    results = {}

    for held_out in CASE_IDS:
        fold = get_fold(held_out)
        train_rows = fold['train']

        mean, std, n = compute_normalization_stats(train_rows['cache_path'].tolist(), n_mels)
        results[f'case{held_out}'] = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'n_frames': int(n),
        }
        print(f"held out case{held_out}: n_frames={n}, mean[0]={mean[0]:.4f}, std[0]={std[0]:.4f}")

        n_test_normal = (fold["test"]["label"] == "normal").sum()
        n_test_anomaly = (fold["test"]["label"] == "anomaly").sum()
        n_val_normal = (fold["val"]["label"] == "normal").sum()
        n_val_anomaly = (fold["val"]["label"] == "anomaly").sum()
        print(
            f"held out case{held_out}: "
            f"train={len(fold['train'])} IND clips from cases {[c for c in CASE_IDS if c != held_out]}, "
            f"val={len(fold['val'])} IND clips (normal={n_val_normal}, anomaly={n_val_anomaly}), "
            f"test={len(fold['test'])} clips (normal={n_test_normal}, anomaly={n_test_anomaly})"
        )

    out_path = PROJECT_ROOT / 'configs' / 'all_folds_normalisation_stats.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        print(f'saved to {out_path}')

if __name__ == "__main__":
    build_fold_stats()

