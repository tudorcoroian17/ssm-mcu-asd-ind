"""
Derives the log-mel parity tolerance for 04_phase_3_mcu_feature_pipeline.md
§3.4, per that section's own instruction: tie the tolerance to the noise level
at which the trained model's AUC actually starts to move, not to a round
number.

Perturbs UNNORMALIZED log-mel (the quantity a board's CMSIS-DSP chain produces
and the quantity §3.4 diffs against) with i.i.d. Gaussian noise, at
increasing std, for the test-time clips only. The reference set (train_emb)
stays exactly as saved -- in deployment the MCU never recomputes it, so
perturbing it here would test a question nobody is asking.

Run on case 1 only. Cases 2-4 sit at or near an AUC ceiling (findings/130 §8);
sweeping noise there would need a large, unrealistic perturbation before AUC
visibly moves, producing a tolerance that looks generous purely because it was
measured against a ceiling effect.

The "AUC starts to move" threshold is tied to SEED_SD (findings/210), the
established seed-to-seed spread at this same cell -- a noise level that moves
AUC by less than that is not distinguishable from ordinary training noise.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT, load_config_by_name
from src.data.folds import get_fold
from src.eval.embeddings import read_embeddings, get_embeddings
from src.eval.auc_pauc import DISTANCE_HEADS
from src.features.baselines import apply_normalization
from src.models.backbone import SSMBackbone

SEED_SD = 0.0132  # findings/210, from the three-seed default-model measurement in findings/130 §8
HELD_OUT_CASE = 1  # only fold with dynamic range; see module docstring
POOLING = 'mean'
NOISE_STDS = np.concatenate([[0.0], np.logspace(-3, 1.0, 30)])  # 0 is the baseline point
SAFETY_MARGIN = 3.0  # recommended tolerance = knee-point std / this
N_REPEATS = 5  # noise draws averaged per std point

# The four Phase 2 refit configs; tolerance should be the tightest across
# whichever remain live deployment candidates (findings/220 §1).
CONFIGS = {
    'A-selective': ('f4cd557b7e3b.yaml', '16662b29beb3'),
    'A-classic':   ('f2578cb06991.yaml', '238a49a973a3'),
    'B-classic':   ('352f70960ed3.yaml', 'b77482e85dc3'),
    'B-selective': ('b39731b66741.yaml', '086acf0275b8'),
}


def load_test_clips_perturbed(test_rows, mean, std, noise_std, rng):
    """
    Mirrors load_fold_clips (src/features/baselines.py) but perturbs the
    UNNORMALIZED array before normalizing. noise_std=0 reproduces
    load_fold_clips exactly -- used as the self-check in main().
    """
    arrays = [np.load(p) for p in test_rows['cache_path']]
    if noise_std > 0:
        arrays = [a + rng.normal(0, noise_std, size=a.shape).astype(np.float32)
                 for a in arrays]
    return np.stack([apply_normalization(a, mean, std) for a in arrays])


def sweep_config(config_name, model_hash, device, rng, head):
    cfg = load_config_by_name(config_name)
    base_dir = PROJECT_ROOT / 'runs' / f'case{HELD_OUT_CASE}' / model_hash

    train_emb, _, _, test_emb_saved, test_labels, _, mean, std = \
        read_embeddings(HELD_OUT_CASE, cfg, POOLING)

    fold = get_fold(HELD_OUT_CASE)
    test_rows = fold['test']
    assert len(test_rows) == len(test_labels), \
        'fold[test] row count does not match the saved test_labels -- get_fold() is not reproducing the split that produced the saved embeddings'

    vals = [np.load(p) for p in test_rows['cache_path']]
    all_vals = np.concatenate([v.ravel() for v in vals])
    print(f'log-mel per-bin std across sample clips: {all_vals.std():.4f}')
    print(f'log-mel range: [{all_vals.min():.3f}, {all_vals.max():.3f}]')

    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.pooling = POOLING

    head_fn = DISTANCE_HEADS[head]
    rows = []
    for noise_std in NOISE_STDS:
        aucs = []
        for _ in range(N_REPEATS if noise_std > 0 else 1):  # std=0 is deterministic, no need to repeat
            X_test = load_test_clips_perturbed(test_rows, mean, std, noise_std, rng)
            test_emb = get_embeddings(model, X_test, device)
            if noise_std == 0.0:
                max_diff = np.abs(test_emb - test_emb_saved).max()
                assert max_diff < 1e-4, f'noise_std=0 does not reproduce saved test_emb (max diff {max_diff:.2e})'
            scores, _ = head_fn(train_emb, test_emb, cfg['seed'])
            aucs.append(roc_auc_score(test_labels, scores))
        rows.append({'noise_std': noise_std, 'auc': np.mean(aucs), 'auc_std': np.std(aucs)})

    df = pd.DataFrame(rows)
    baseline_auc = df.loc[df.noise_std == 0.0, 'auc'].iloc[0]
    df['auc_drop'] = baseline_auc - df['auc']
    return df, baseline_auc


def find_knee(df, min_hold=3):
    """
    Smallest noise_std after which auc_drop stays above SEED_SD for at least
    min_hold consecutive points AND all remaining points. Guards against a
    single edge-of-sweep-range point being read as a sustained knee -- e.g. a
    config that improves under noise for the entire tested range and then
    crashes only at the very last point tested has no real knee, just an
    unexplored cliff.
    """
    drops = df.sort_values('noise_std')['auc_drop'].values
    stds = df.sort_values('noise_std')['noise_std'].values
    for i in range(len(drops) - min_hold + 1):
        if np.all(drops[i:] > SEED_SD):
            return float(stds[i])
    return None


def main(head):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(158)

    results = {}
    for label, (config_name, model_hash) in CONFIGS.items():
        print(f'\n=== {label} ({config_name}) ===')
        df, baseline_auc = sweep_config(config_name, model_hash, device, rng, head)
        print(df.to_string(index=False))
        knee = find_knee(df)
        results[label] = {'df': df, 'baseline_auc': baseline_auc, 'knee': knee}
        print(f'baseline AUC={baseline_auc:.4f}  knee (drop > {SEED_SD})='
              f'{knee if knee else "not reached in sweep range"}')

    knees = [r['knee'] for r in results.values() if r['knee'] is not None]
    if knees:
        tightest = min(knees)
        tolerance = tightest / SAFETY_MARGIN
        print(f'\ntightest knee across configs: {tightest:.4f}')
        print(f'recommended per-bin log-mel std tolerance '
              f'(knee / {SAFETY_MARGIN}x safety margin): {tolerance:.4f}')
    else:
        print('\nno config reached the knee within the sweep range -- widen NOISE_STDS')

    return results


if __name__ == '__main__':
    heads = ['knn_clustered_16', 'euclidean']
    for h in heads:
        print(f'\n=== Running for head ({h}) ===')
        main(h)