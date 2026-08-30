"""
Three threshold methods per 01_eval_spec.md §6 and master doc §9.1:
  percentile   -- from val-normal scores, purely unsupervised. Primary.
  chi_square   -- analytic, Mahalanobis only (squared distance ~ chi2(d_model)
                  under a Gaussian-normal assumption). Cross-check.
  calibrated   -- sweeps threshold against val-normal + val-anomaly to hit
                  best F1. Upper bound; uses label information a real
                  deployment wouldn't have. Flag this in any write-up.

Secondary metrics (precision, recall, accuracy, F1) are reported per
threshold method, per 01_eval_spec.md §6 -- never as one unlabeled set.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from runs.compute_hash import train_config_hash
from src.config import load_config, PROJECT_ROOT
from src.eval.embeddings import read_embeddings
from src.eval.auc_pauc import (
    DISTANCE_HEADS, score_euclidean, score_mahalanobis, score_knn_full, score_knn_clustered,
)

POOLING_MODES = ['mean', 'max', 'concat_mean_last']

def secondary_metrics(scores, labels, threshold):
    pred = (scores > threshold).astype(int)
    return {
        'threshold': float(threshold),
        'precision': float(precision_score(labels, pred, zero_division=0)),
        'recall': float(recall_score(labels, pred, zero_division=0)),
        'accuracy': float(accuracy_score(labels, pred)),
        'f1': float(f1_score(labels, pred, zero_division=0)),
    }

def percentile_threshold(normal_scores, percentile=95.0):
    return np.percentile(normal_scores, percentile)

def chi_square_threshold(d_model, alpha=0.01):
    # squared Mahalanobis distance ~ chi2(d_model) under a Gaussian-normal
    # assumption -- score_mahalanobis() returns the SQUARE ROOT, so undo that
    return float(np.sqrt(chi2.ppf(1 - alpha, df=d_model)))

def calibrated_threshold(val_normal_scores, val_anomaly_scores):
    # Needs normal and anomaly scores to compute threshold. Not applicable for on_held_out
    # threshold computation
    val_scores = np.concatenate([val_normal_scores, val_anomaly_scores])
    val_labels = np.concatenate([np.zeros(len(val_normal_scores)), np.ones(len(val_anomaly_scores))])
    candidates = np.unique(val_scores)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = f1_score(val_labels, (val_scores > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)

def run_case(held_out_case, cfg, percentile=95, chi2_alpha=0.01, on_held_out=False):
    dir_name = train_config_hash(cfg, held_out_case)
    out_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name
    rows = []

    for pooling_mode in POOLING_MODES:
        train_emb, val_normal_emb, val_anomaly_emb, test_emb, test_labels, calib_normal, _, _ = \
            read_embeddings(held_out_case, cfg, pooling_mode)

        for head_name, head_fn in DISTANCE_HEADS.items():
            test_scores, extra = head_fn(train_emb, test_emb, cfg['seed'])

            methods = {}

            if on_held_out:
                # Compute threshold bases on the calib_normal set (same case as held_out_case)
                calib_normal_scores, _ = head_fn(train_emb, calib_normal, cfg['seed'])
                for pct in [95.0, 99.0, 99.5]:
                    methods[f'perentile_same_machie_{pct}'] = percentile_threshold(calib_normal_scores, pct)
            else:
                # Compute threshold based on validation set (cases other than held_out_case)
                val_normal_scores, _ = head_fn(train_emb, val_normal_emb, cfg['seed'])
                val_anomaly_scores, _ = head_fn(train_emb, val_anomaly_emb, cfg['seed'])
                methods['percentile'] = percentile_threshold(val_normal_scores, percentile)
                if head_name == 'mahalanobis':
                    methods['chi_square'] = chi_square_threshold(train_emb.shape[1], chi2_alpha)
                methods['calibrated'] = calibrated_threshold(val_normal_scores, val_anomaly_scores)

            for method_name, t in methods.items():
                m = secondary_metrics(test_scores, test_labels, t)
                rows.append({
                    'held_out_case': held_out_case, 'pooling': pooling_mode,
                    'distance_head': head_name, 'threshold_method': method_name, **m,
                })

    if on_held_out:
        out_fn_stem = out_dir / 'thresholds_same_machine'
    else:
        out_fn_stem = out_dir / 'threshold_metrics'

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'{out_fn_stem}.csv', index=False)
    with open(out_dir / f'{out_fn_stem}.json', 'w') as f:
        json.dump(rows, f, indent=4)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--same_machine', action='store_true')
    args = parser.parse_args()

    cfg = load_config()
    for case in [1, 2, 3, 4]:
        print(f'\n=== held_out_case {case} ===')
        rows = run_case(case, cfg, on_held_out=args.same_machine)
        for r in rows:
            print(f"{r['pooling']:20s} {r['distance_head']:20s} {r['threshold_method']:12s} "
                  f"P={r['precision']:.3f} R={r['recall']:.3f} A={r['accuracy']:.3f} F1={r['f1']:.3f}")