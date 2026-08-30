"""
Diagnoses whether threshold failure is a calibration-transfer problem or a
ranking problem. Prints, per pooling x head:
  - where val-normal scores sit vs where test scores sit (the shift itself)
  - F1 from a val-calibrated threshold vs F1 from the best threshold on test
The gap between those two F1 values IS the calibration transfer loss.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.config import load_config
from src.eval.embeddings import read_embeddings
from src.eval.auc_pauc import DISTANCE_HEADS
from src.eval.thresholds import percentile_threshold, calibrated_threshold

POOLING_MODES = ['mean', 'max', 'concat_mean_last']


def best_f1_on(scores, labels):
    """Oracle threshold: best achievable F1 if the scale were perfectly known."""
    best_t, best = scores[0], -1.0
    for t in np.unique(scores):
        f = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, t
    return best, best_t

def per_machine_percentile_threshold(test_normal_scores, percentile=95):
    """
    Fourth method, for reporting only. Simulates real deployment: the unit
    records its own normal operation after installation and thresholds from
    that. Uses held-out-case data, so it is NOT unsupervised-clean under the
    LOSO protocol -- flag it at least as loudly as method 3. It is, however,
    substantially more deployment-realistic than method 3, and unlike the
    oracle it never touches anomaly labels.
    """
    return np.percentile(test_normal_scores, percentile)


def run(held_out_case, cfg):
    print(f"\n{'=' * 104}\ncase{held_out_case}")
    for pooling in POOLING_MODES:
        train_emb, val_n_emb, val_a_emb, test_emb, test_labels, _, _ = \
            read_embeddings(held_out_case, cfg, pooling)

        for head_name, head_fn in DISTANCE_HEADS.items():
            test_s, _ = head_fn(train_emb, test_emb, cfg['seed'])
            val_n_s, _ = head_fn(train_emb, val_n_emb, cfg['seed'])
            val_a_s, _ = head_fn(train_emb, val_a_emb, cfg['seed'])

            test_norm = test_s[test_labels == 0]
            test_anom = test_s[test_labels == 1]

            # Does ANY test clip score below the val-normal bulk?
            frac_below_p95 = (test_norm < np.percentile(val_n_s, 95)).mean()

            t_pct = percentile_threshold(val_n_s, 95)
            t_cal = calibrated_threshold(val_n_s, val_a_s)
            f1_pct = f1_score(test_labels, (test_s >= t_pct).astype(int), zero_division=0)
            f1_cal = f1_score(test_labels, (test_s >= t_cal).astype(int), zero_division=0)
            f1_oracle, t_oracle = best_f1_on(test_s, test_labels)
            auc = roc_auc_score(test_labels, test_s)
            pct_same_held_out_case = per_machine_percentile_threshold(test_norm, 95)

            print(f"\n  {pooling} / {head_name}   AUC={auc:.4f}")
            print(f"    val_normal   p50={np.median(val_n_s):8.3f}  p95={np.percentile(val_n_s, 95):8.3f}  max={val_n_s.max():8.3f}")
            print(f"    test_normal  p05={np.percentile(test_norm, 5):8.3f}  p50={np.median(test_norm):8.3f}  min={test_norm.min():8.3f}")
            print(f"    test_anomaly p05={np.percentile(test_anom, 5):8.3f}  p50={np.median(test_anom):8.3f}  min={test_anom.min():8.3f}")
            print(f"    test normals below val-normal p95: {frac_below_p95:.1%}")
            print(f"    test normals percentile (from held_out_case): {pct_same_held_out_case:.3f}")
            print(f"    F1  percentile={f1_pct:.3f}  calibrated={f1_cal:.3f}  "
                  f"ORACLE={f1_oracle:.3f} (t={t_oracle:.3f})   "
                  f"transfer loss = {f1_oracle - max(f1_pct, f1_cal):.3f}")


if __name__ == '__main__':
    cfg = load_config()
    for case in [1, 2, 3, 4]:
        run(case, cfg)