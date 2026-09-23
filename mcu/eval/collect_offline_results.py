import json
from argparse import ArgumentParser

import pandas as pd

from src.config import PROJECT_ROOT

MCU_DEPLOY_DIR = PROJECT_ROOT / 'mcu' / 'deploy'

def get_metric_rows(model_dir, is_q15_feat = False, mode = 'dropin'):
    first_metrics_rows = []
    second_metrics_rows = []
    setups_dir = model_dir / 'setups'
    for setup in setups_dir.iterdir():
        if not setup.is_dir():
            continue
        diagnostics_file = setup / 'diagnostics.json'
        offline_dir = 'offline'
        if is_q15_feat:
            offline_dir += '_q15'
            if mode == 'dropin':
                offline_dir += '_dropin'
            else:
                offline_dir += '_recal'
        metrics_file = setup / offline_dir / 'metrics.json'
        with open(diagnostics_file) as f:
            diag = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)
        h_width = diag['backbone'].get('h_width', 'N/A')
        weights = diag['backbone'].get('weight_mode', 'N/A')
        granularity = diag['backbone'].get('granularity', 'N/A')
        activation_group = diag['backbone'].get('activation_group', 'N/A')
        recurrence = diag['backbone'].get('recurrence', 'N/A')
        fm_row = (
            diag['config'],
            diag['held_out_case'],
            diag['model_hash'],
            diag['identifier'],
            diag['combo_number'],
            diag['backbone']['family'],
            h_width,
            weights,
            granularity,
            activation_group,
            recurrence,
            diag['head'],
            diag['head_precision'],
            metrics['auc'],
            metrics['pauc'],
        )
        sm_row = (
            diag['config'],
            diag['held_out_case'],
            diag['model_hash'],
            diag['identifier'],
            diag['combo_number'],
            diag['backbone']['family'],
            h_width,
            weights,
            granularity,
            activation_group,
            recurrence,
            diag['head'],
            diag['head_precision'],
        )
        for k,v in metrics['thresholds'].items():
            temp = sm_row + (k, v['metrics']['precision'], v['metrics']['recall'], v['metrics']['accuracy'], v['metrics']['f1'])
            second_metrics_rows.append(temp)
        first_metrics_rows.append(fm_row)

    return first_metrics_rows, second_metrics_rows

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--held-out-case', type=int, required=True)
    parser.add_argument('--quant-features', action='store_true')
    parser.add_argument('--mode', type=str, default='dropin', help='q15 mode can be dropin or recal')
    args = parser.parse_args()

    if args.quant_features:
        out_subdir = 'q15_features'
    else:
        out_subdir = 'fp32_features'

    case_dir = MCU_DEPLOY_DIR / f'case{args.held_out_case}'
    out_dir = PROJECT_ROOT / 'mcu' / 'eval' / 'offline' / out_subdir
    columns_fm = ['config', 'held_out_case', 'model_hash', 'identifier', 'combo_number',
                  'bb_family', 'bb_h_width', 'bb_weights', 'bb_granularity', 'bb_activation_group', 'bb_recurrence',
                  'head', 'head_precision', 'auc', 'pauc']
    columns_sm = ['config', 'held_out_case', 'model_hash', 'identifier', 'combo_number',
                  'bb_family', 'bb_h_width', 'bb_weights', 'bb_granularity', 'bb_activation_group', 'bb_recurrence',
                  'head', 'head_precision',
                  'threshold_method', 'precision', 'recall', 'accuracy', 'f1']
    fm_rows = []
    sm_rows = []

    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        fmrs, smrs = get_metric_rows(entry)
        fm_rows.extend(fmrs)
        sm_rows.extend(smrs)

    first_metrics_df = pd.DataFrame.from_records(fm_rows, columns=columns_fm)
    second_metrics_df = pd.DataFrame.from_records(sm_rows, columns=columns_sm)
    first_metrics_df.to_csv(out_dir / 'first_metrics.csv', index=False)
    second_metrics_df.to_csv(out_dir / 'second_metrics.csv', index=False)
