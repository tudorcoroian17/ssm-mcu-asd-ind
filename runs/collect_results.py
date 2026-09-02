"""
Walks every trained run and collapses its artifacts into four tables.

Four, not one, because the metrics live at different grains: training numbers
are per config, footprint is per pooling, half-life is per layer, zero-state is
per batch, and the threshold metrics are per pooling x head x method. Every
value is carried at its native grain and nothing is aggregated here, so no
number in these tables is a summary of another number.

  configs.csv      one row per (config, case)
  footprints.csv   one row per (config, case, pooling)
  diagnostics.csv  one row per (config, case, check, unit, metric)
  scores.csv       one row per (config, case, pooling, head, threshold_method)

Only same-machine threshold calibration is collected. threshold_metrics.csv
predates that change and is deliberately skipped.
"""
import json

import pandas as pd

from src.config import PROJECT_ROOT

CONFIGS_ROOT = PROJECT_ROOT / 'configs' / 'ablation'
RUNS_ROOT = PROJECT_ROOT / 'runs'
OUT_ROOT = RUNS_ROOT / 'phase2'
KNOBS = ['d_state', 'n_layers', 'expand', 'selective', 'discretization']

_n_train_cache = {}


def n_train_embeddings(base_dir, case):
    """
    Reference-set size, read from the embeddings manifest rather than assumed.
    Depends only on the fold, so cache it per case.
    """
    if case not in _n_train_cache:
        man = pd.read_csv(base_dir / 'embeddings' / 'manifest.csv')
        n = int((man['used_in'] == 'training').sum())
        # A silent zero here prices knn_full's 810 KB reference set at nothing
        # and inverts the entire Pareto frontier. Fail loudly instead.
        assert n > 0, (f"no rows with used_in == 'training' in {base_dir}; "
                       f"labels present: {man['used_in'].unique().tolist()}")
        _n_train_cache[case] = n
    return _n_train_cache[case]


def head_flash_fp32_kb(head, pooling, n_train, d_model=64):
    """
    Distance-head storage, in KB. footprint.csv covers the backbone only, and
    knn_full stores the whole training set: at 3240 x 64 floats that is 810 KB,
    over five times the 151.8 KB backbone (findings/130 section 11). A Pareto
    plot without this ranks knn_full first on a number that ignores it.
    """
    d_emb = 2 * d_model if pooling == 'concat_mean_last' else d_model
    n_floats = {
        'euclidean': d_emb,
        'knn_clustered_16': 16 * d_emb,
        'mahalanobis': d_emb + d_emb ** 2,
        'knn_full': n_train * d_emb,
    }[head]
    return n_floats * 4 / 1024


def collect():
    manifest = pd.read_csv(CONFIGS_ROOT / '000_config_manifest.csv')
    sweep_log_path = RUNS_ROOT / 'sweep_logs.csv'
    sweep_log = pd.read_csv(sweep_log_path) if sweep_log_path.exists() else None

    config_rows, footprint_frames, diag_rows, score_frames = [], [], [], []
    missing = []

    for _, row in manifest.iterrows():
        case, model_hash = int(row['held_out_case']), row['model_hash']
        base_dir = RUNS_ROOT / f'case{case}' / model_hash
        if not (base_dir / 'ckpt.pt').exists():
            continue

        key = {'config_name': row['config_name'], 'model_hash': model_hash,
               'held_out_case': case}
        knobs = {k: row[k] for k in KNOBS}

        # --- config grain -------------------------------------------------
        cfg_row = {**key, **knobs}
        if sweep_log is not None:
            hit = sweep_log[(sweep_log['model_hash'] == model_hash)
                            & (sweep_log['held_out_case'] == case)]
            if len(hit):
                last = hit.iloc[-1]   # a re-run appends; the last row is current
                cfg_row.update({c: last[c] for c in
                                ['status', 'train_skill', 'val_skill',
                                 'best_val_mse', 'epochs_run', 'elapsed_s']})
        config_rows.append(cfg_row)

        # --- footprint grain: one row per pooling, kept as written ---------
        fp_path = base_dir / 'footprint.csv'
        fp = None
        if fp_path.exists():
            fp = pd.read_csv(fp_path)
            fp_out = fp.copy()
            for k, v in {**key, **knobs}.items():
                fp_out[k] = v
            footprint_frames.append(fp_out)
        else:
            missing.append((model_hash, case, 'footprint.csv'))

        # --- diagnostics, long --------------------------------------------
        zs_path = base_dir / 'zero_state.csv'
        if zs_path.exists():
            zs = pd.read_csv(zs_path)
            # The recurrence's contribution beyond what the depthwise conv
            # already supplies. Near zero means a decorative scan.
            zs['state_contribution'] = zs['skill_real'] - zs['skill_zeroed']
            for _, z in zs.iterrows():
                for metric in ['real_loss', 'zeroed_loss', 'persistence',
                               'skill_real', 'skill_zeroed', 'state_contribution']:
                    diag_rows.append({**key, **knobs, 'check': 'zero_state',
                                      'unit': z['limits'], 'metric': metric,
                                      'value': z[metric]})

        hl_path = base_dir / 'decay_half_life.csv'
        if hl_path.exists():
            hl = pd.read_csv(hl_path)
            metrics = [c for c in ['min_ms', 'max_ms', 'median_ms', 'mean_ms',
                                   'std_ms', 'frac_below_conv_window'] if c in hl]
            for _, h in hl.iterrows():
                for metric in metrics:
                    diag_rows.append({**key, **knobs, 'check': 'half_life',
                                      'unit': f"layer{int(h['layer'])}",
                                      'metric': metric, 'value': h[metric]})

        # --- score grain ---------------------------------------------------
        scores_path = base_dir / 'eval' / 'results.json'
        thr_path = base_dir / 'thresholds_same_machine.csv'
        if not scores_path.exists() or not thr_path.exists():
            missing.append((model_hash, case, 'eval/results.json or thresholds'))
            continue

        # JSON rather than results.csv: cov_condition_number is only in the JSON.
        scores = pd.DataFrame(json.load(open(scores_path)))
        keep = ['pooling', 'distance_head', 'auc', 'pauc']
        if 'cov_condition_number' in scores:
            keep.append('cov_condition_number')

        merged = pd.read_csv(thr_path).merge(
            scores[keep], on=['pooling', 'distance_head'], how='left')
        for k, v in {**key, **knobs}.items():
            merged[k] = v

        if fp is not None:
            merged = merged.merge(
                fp[['pooling', 'flash_footprint_int8_kb',
                    'streaming_peak_ram_int8_kb', 'streaming_peak_ram_fp32_kb']],
                on='pooling', how='left')
            n_train = n_train_embeddings(base_dir, case)
            merged['head_flash_fp32_kb'] = [
                head_flash_fp32_kb(h, p, n_train)
                for h, p in zip(merged['distance_head'], merged['pooling'])]
            merged['total_flash_kb'] = (merged['flash_footprint_int8_kb']
                                        + merged['head_flash_fp32_kb'])

        # Degenerate thresholds, flagged per row and never dropped here.
        # evt_p999_equiv on max+euclidean sits above every score: precision 0,
        # recall 0, accuracy 0.5. That is threshold placement failing, not model
        # quality, and it must be visible rather than silently included later.
        merged['flags_nothing'] = merged['recall'] < 0.01
        merged['flags_everything'] = (merged['recall'] > 0.99) & (merged['precision'] < 0.55)

        score_frames.append(merged)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(config_rows).to_csv(OUT_ROOT / 'configs.csv', index=False)
    pd.concat(footprint_frames, ignore_index=True).to_csv(
        OUT_ROOT / 'footprints.csv', index=False)
    pd.DataFrame(diag_rows).to_csv(OUT_ROOT / 'diagnostics.csv', index=False)
    pd.concat(score_frames, ignore_index=True).to_csv(
        OUT_ROOT / 'scores.csv', index=False)

    print(f'configs.csv     {len(config_rows)} rows')
    print(f'footprints.csv  {sum(len(f) for f in footprint_frames)} rows')
    print(f'diagnostics.csv {len(diag_rows)} rows')
    print(f'scores.csv      {sum(len(f) for f in score_frames)} rows')
    if missing:
        print(f'\n{len(missing)} missing artifacts:')
        for m in missing[:20]:
            print('  ', m)


if __name__ == '__main__':
    collect()