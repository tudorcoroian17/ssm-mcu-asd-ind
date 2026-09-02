import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from scipy.stats import genpareto, gamma, lognorm, weibull_min, chi as chi_dist
from scipy.stats import gaussian_kde

from runs.compute_hash import train_config_hash
from src.config import load_config, PROJECT_ROOT, load_config_by_name
from src.eval.embeddings import read_embeddings
from src.eval.auc_pauc import DISTANCE_HEADS

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

# --- 1. Extreme Value Theory (peak-over-threshold, GPD tail) -------------
def evt_threshold(calib_normal_scores, target_far=0.05, tail_fraction=0.10):
    """
    POT/GPD threshold (Coles 2001 sec 4.3; Siffer et al. 2017 SPOT). Fits a
    2-parameter Generalized Pareto Distribution to the exceedances over an
    initial threshold u0, then extrapolates to the quantile corresponding
    to target_far.

    Directly targets the p99.5 collapse seen in
    checks/metrics/threshold_transfer.py (case3, mean/euclidean, F1=0.171):
    that threshold was set by ~5 order statistics. EVT uses every
    exceedance above u0 (~n*tail_fraction points) to fit a smooth tail
    shape instead, and remains meaningful even at target_far below what
    the sample size could estimate empirically.
    """
    x = np.sort(np.asarray(calib_normal_scores))
    n = len(x)
    n_u = max(20, int(n * tail_fraction))
    u0 = x[-n_u - 1]
    excesses = x[-n_u:] - u0
    excesses = excesses[excesses > 0]

    try:
        shape, _, scale = genpareto.fit(excesses, floc=0)
    except Exception:
        shape, scale = 0.0, float(np.mean(excesses))  # exponential-tail fallback

    zeta_u = n_u / n
    ratio = target_far / zeta_u

    if ratio >= 1:
        # target_far isn't actually in the tail region -- extrapolation
        # isn't valid; fall back to the empirical quantile.
        return float(percentile_threshold(x, (1 - target_far) * 100)), {
            'u0': float(u0), 'shape': None, 'scale': None, 'n_exceedances': n_u,
            'note': 'target_far above tail region, used empirical percentile',
        }

    if abs(shape) < 1e-6:
        z_q = u0 + scale * (-np.log(ratio))
    else:
        z_q = u0 + (scale / shape) * (ratio ** (-shape) - 1)

    return float(z_q), {'u0': float(u0), 'shape': float(shape), 'scale': float(scale), 'n_exceedances': n_u}

# --- 2. Parametric / heavy-tail distribution fitting ---------------------
PARAMETRIC_CANDIDATES = {
    'gamma': gamma,
    'lognorm': lognorm,
    'weibull_min': weibull_min,
    'chi': chi_dist,   # natural family for Euclidean distance -- the
                        # chi-square precedent (01_eval_spec.md sec 6) is
                        # for squared Mahalanobis; chi is its analogue for
                        # a raw distance under a Gaussian-isotropic cluster
}

def parametric_threshold(calib_normal_scores, target_far=0.05, candidates=PARAMETRIC_CANDIDATES):
    """
    Fits each candidate family via MLE, selects the best by AIC, returns
    the (1 - target_far) quantile of the winning fit.
    """
    x = np.asarray(calib_normal_scores)
    x = x[x > 0]

    results = {}
    for name, dist in candidates.items():
        try:
            params = dist.fit(x, floc=0)
            loglik = np.sum(dist.logpdf(x, *params))
            aic = 2 * len(params) - 2 * loglik
            results[name] = {'params': params, 'aic': aic}
        except Exception as e:
            results[name] = {'error': str(e)}

    valid = {k: v for k, v in results.items() if 'aic' in v}
    if not valid:
        raise RuntimeError('No parametric family converged')
    best_name = min(valid, key=lambda k: valid[k]['aic'])
    best = valid[best_name]
    threshold = float(candidates[best_name].ppf(1 - target_far, *best['params']))

    return threshold, {
        'best_family': best_name,
        'aic_by_family': {k: v.get('aic') for k, v in results.items()},
    }

# --- 3. Kernel Density Estimation -----------------------------------------
def kde_threshold(calib_normal_scores, target_far=0.05, n_resample=200_000, seed=None):
    """
    Non-parametric threshold: Gaussian KDE fit to calibration normals,
    quantile taken from a large resample of the fitted density. No assumed
    shape, smoother than a raw percentile at small n.
    """
    kde = gaussian_kde(np.asarray(calib_normal_scores))
    rng = np.random.default_rng(seed)
    resampled = kde.resample(n_resample, seed=rng)[0]
    resampled = resampled[resampled > 0]
    threshold = float(np.percentile(resampled, (1 - target_far) * 100))
    return threshold, {'bandwidth_factor': float(kde.factor), 'n_calib': len(calib_normal_scores)}

# --- 4. Robust dispersion metrics -----------------------------------------
def mad_threshold(calib_normal_scores, k=3.5):
    """
    Modified z-score rule (Iglewicz & Hoaglin 1993), k=3.5 is the standard
    outlier cutoff. Doesn't target an exact false-alarm rate -- uses only
    the median and MAD, both 50%-breakdown-point robust statistics.
    """
    x = np.asarray(calib_normal_scores)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    threshold = median + (k / 0.6745) * mad
    return float(threshold), {'median': float(median), 'mad': float(mad)}

def iqr_threshold(calib_normal_scores, k=3.0):
    """
    Tukey fence. k=3.0 ("far out") rather than the conventional 1.5
    ("mild outlier") -- 1.5 would flag too large a fraction of normal
    clips for an anomaly threshold.
    """
    x = np.asarray(calib_normal_scores)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    threshold = q3 + k * iqr
    return float(threshold), {'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr)}

# --- 5. False alarm budgeting --------------------------------------------
def target_far_from_budget(false_alarms_per_period, clips_per_period):
    """
    Converts a deployment false-alarm budget into the target_far parameter
    consumed by evt_threshold(), parametric_threshold(), and kde_threshold().

    Example: "at most 1 false alarm per week", machine scored once every
    5 minutes -> clips_per_period = 7*24*12 = 2016 -> target_far = 0.000496.

    Placeholder cadence -- ToyADMOS2's IND clips are individual motor
    cycles from the dataset's own recording protocol, not a real sensor's
    sampling rate. Plug in the actual deployment duty cycle.
    """
    return false_alarms_per_period / clips_per_period


FAR_BUDGETS = {
    'p95_equiv': 0.05,     # ~1 false alarm per 20 cycles -- lenient, matches current p95 baseline
    'p99_equiv': 0.01,     # ~1 per 100 cycles
    'p999_equiv': 0.001,   # ~1 per 1000 cycles -- below where plain percentiles are estimable at n~1000
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

def run_case(held_out_case, cfg, dir_name, percentile=95, chi2_alpha=0.01, on_held_out=False):
    out_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name
    rows, diagnostics = [], []

    for pooling_mode in POOLING_MODES:
        train_emb, val_normal_emb, val_anomaly_emb, test_emb, test_labels, calib_normal, _, _ = \
            read_embeddings(held_out_case, cfg, pooling_mode)

        for head_name, head_fn in DISTANCE_HEADS.items():
            test_scores, extra = head_fn(train_emb, test_emb, cfg['seed'])

            methods, info = {}, {}

            if on_held_out:
                # Compute threshold bases on the calib_normal set (same case as held_out_case)
                calib_normal_scores, _ = head_fn(train_emb, calib_normal, cfg['seed'])
                for pct in [95.0, 99.0, 99.5]:
                    methods[f'perentile_same_machie_{pct}'] = percentile_threshold(calib_normal_scores, pct)
                for far_name, far in FAR_BUDGETS.items():
                    methods[f'evt_{far_name}'], info[f'evt_{far_name}'] = evt_threshold(calib_normal_scores, far)
                    methods[f'parametric_{far_name}'], info[f'parametric_{far_name}'] = parametric_threshold(
                        calib_normal_scores, far)
                    methods[f'kde_{far_name}'], info[f'kde_{far_name}'] = kde_threshold(calib_normal_scores, far,seed=cfg['seed'])
                methods['mad'], info['mad'] = mad_threshold(calib_normal_scores)
                methods['iqr'], info['iqr'] = iqr_threshold(calib_normal_scores)
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

            diagnostics.append({
                'pooling': pooling_mode, 'distance_head': head_name, 'info': info,
            })

    if on_held_out:
        out_fn_stem = out_dir / 'thresholds_same_machine'
    else:
        out_fn_stem = out_dir / 'threshold_metrics'

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'{out_fn_stem}.csv', index=False)
    with open(out_dir / f'{out_fn_stem}.json', 'w') as f:
        json.dump(rows, f, indent=4)
    with open(out_dir / 'threshold_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=4, default=str)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--same_machine', action='store_true')
    args = parser.parse_args()

    configs_root = PROJECT_ROOT / 'configs' / 'ablation'
    runs_root = PROJECT_ROOT / 'runs'
    configs_manifest = pd.read_csv(configs_root / '000_config_manifest.csv')

    for index, row in configs_manifest.iterrows():
        case = int(row['held_out_case'])
        model_hash = row['model_hash']
        ckpt_path = runs_root / f'case{case}' / model_hash / 'ckpt.pt'

        if not ckpt_path.exists():
            print(f'model {model_hash} not cached')
            continue

        config_file = load_config_by_name(row['config_name'])

        print(f'\n=== generating embeddings for case {case} -> model {model_hash} ===')
        rows = run_case(case, config_file, model_hash, on_held_out=args.same_machine)
        for r in rows:
            print(f"{r['pooling']:20s} {r['distance_head']:20s} {r['threshold_method']:12s} "
                  f"P={r['precision']:.3f} R={r['recall']:.3f} A={r['accuracy']:.3f} F1={r['f1']:.3f}")
