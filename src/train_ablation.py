import json
import time
import random
import traceback

import pandas as pd
import torch
import hashlib
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import PROJECT_ROOT, load_config, load_config_by_name
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips, compute_baselines
from src.models.backbone import SSMBackbone
from src.models.heads import PredictionHead
from runs.compute_hash import train_config_hash
from src.train import ClipDataset, set_seed, compute_loss

def train_one_fold(held_out_case, cfg, dir_name):
    local_configs = {
        'held_out_case': held_out_case,
        'training': cfg['training'],
        'features': cfg['features'],
        'model': cfg['model'],
    }
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_mode = cfg['training']['target']

    fold = get_fold(held_out_case)
    k = cfg['training']['horizon_k']
    n_mels = cfg['features']['n_mels']

    mean, std, n = compute_normalization_stats(fold['train']['cache_path'], n_mels)

    X_train = load_fold_clips(fold['train'], mean, std)

    # Early stopping: never touches the held-out case; normal-only —
    # anomalies in val are reserved for threshold calibration
    val_normal_rows = fold['val'][fold['val']['label'] == 'normal']
    X_val = load_fold_clips(val_normal_rows, mean, std)

    train_loader = DataLoader(ClipDataset(X_train), batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(ClipDataset(X_val), batch_size=cfg['training']['batch_size'], shuffle=False)

    model = SSMBackbone(**cfg['model']).to(device)
    head = PredictionHead(cfg['model']['d_model'], cfg['model']['n_mels']).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg['training']['learning_rate'],
    )

    best_val_mse = float('inf')
    min_val_improv_delta = cfg['training'].get('min_val_improv_delta', 3e-4)
    epochs_without_improvement = 0
    best_state = None
    final_train_mse = None

    train_baseline = compute_baselines(X_train, k=k)
    val_baseline = compute_baselines(X_val, k=k)

    all_train_mse = []
    all_val_mse = []

    for epoch in range(cfg['training']['max_epochs']):
        model.train()
        head.train()
        train_losses = []

        t0 = time.time()
        for x in tqdm(train_loader, desc=f'epoch {epoch}'):
            x = x.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, head, x, k, target_mode=target_mode)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        final_train_mse = float(np.mean(train_losses))
        print(f"epoch {epoch} train pass: {time.time() - t0:.1f}s")

        model.eval()
        head.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                val_losses.append(compute_loss(model, head, x, k, target_mode=target_mode).item())
        val_mse = float(np.mean(val_losses))

        print(f"epoch {epoch}: train_mse={final_train_mse:.5f}  |  val_mse={val_mse:.5f}  "
              f"|  best_val_mse={best_val_mse:.5f}  |  epochs_without_improvement={epochs_without_improvement}")
        all_train_mse.append(final_train_mse)
        all_val_mse.append(val_mse)

        if val_mse < best_val_mse - min_val_improv_delta:
            best_val_mse = val_mse
            epochs_without_improvement = 0
            best_state = {'model': model.state_dict(), 'head': head.state_dict()}
            ckpt_path = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name /f'ckpt.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)

            ckpt_descriptor_path = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name / f'ckpt_descriptor.json'
            ckpt_descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            local_configs['held_out_case'] = held_out_case
            local_configs['val_mse'] = val_mse
            local_configs['train_mse'] = final_train_mse
            local_configs['all_train_mse'] = all_train_mse
            local_configs['all_val_mse'] = all_val_mse
            local_configs['train_skill'] = 1 - final_train_mse / train_baseline['mse_persistence']
            local_configs['val_skill'] = 1 - val_mse / val_baseline['mse_persistence']
            with open(ckpt_descriptor_path, 'w') as f:
                json.dump(local_configs, f, indent=4)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg['training']['early_stopping_patience']:
                print(f"early stopping at epoch {epoch}")
                break

    return {
        'held_out_case': held_out_case,
        'best_val_mse': best_val_mse,
        'final_train_mse': final_train_mse,
        'train_baseline': train_baseline,
        'val_baseline': val_baseline,
        'all_train_mse': all_train_mse,
        'all_val_mse': all_val_mse,
        'model_state': best_state['model'],
        'head_state': best_state['head'],
    }

def sanity_gate(res):
    train_skill = 1 - res['final_train_mse'] / res['train_baseline']['mse_persistence']
    val_baseline_mse_persistence = res['val_baseline']['mse_persistence']
    val_skill = 1 - res['best_val_mse'] / val_baseline_mse_persistence
    print(f'train skill: {train_skill:.4f}  |  '
          f'val skill: {val_skill:.4f}  |  '
          f'mse_persistence val_baseline: {val_baseline_mse_persistence}')

    if val_skill <= 0:
        print("FAILED sanity gate — val skill <= 0. Check, in order:")
        print("  1. k too small — still copyable")
        print("  2. non-causal leak in the conv (ssm_block.py GAP 1)")
        print("  3. learning rate")
        print("  4. normalization mismatch between baseline and training-loss units")
    return train_skill, val_skill

if __name__ == "__main__":
    ORDER_SEED = 158
    configs_root = PROJECT_ROOT / 'configs' / 'ablation'
    runs_root = PROJECT_ROOT / 'runs'
    log_path = runs_root / 'sweep_logs.csv'
    config_manifest = pd.read_csv(str(configs_root / '000_config_manifest.csv'))
    configs_for_train = config_manifest[(config_manifest['target'] == 'residual') &
                                        (config_manifest['horizon_k'] == 2) &
                                        (config_manifest['held_out_case'] == 1)]
    queue = configs_for_train.sample(frac=1.0, random_state=ORDER_SEED)
    print(f'{len(queue)} configs in queue')

    default_cfg = load_config()
    for index, row in queue.iterrows():
        case = int(row['held_out_case'])
        model_hash = row['model_hash']
        ckpt_path = runs_root / f'case{case}' / model_hash / 'ckpt.pt'

        # Disk is the resume state, not a manifest column: a crash between
        # torch.save and a CSV write would desync the two.
        if ckpt_path.exists():
            print(f"{row['config_name']} case{case} -> {model_hash} already cached at {ckpt_path}")
            continue
        config_file = load_config_by_name(str(configs_root / row['config_name']))

        # The manifest supplies the directory name, so a config_name/model_hash
        # mismatch across 288 generated files would silently write one config's
        # run into another's directory.
        assert train_config_hash(config_file, case) == model_hash, (
            f"{row['config_name']} hashes to {train_config_hash(config_file, case)}, "
            f"manifest says {model_hash}")
        # src/data/folds.py and src/features/baselines.py read default.yaml at
        # import time. Correct only while every ablation shares these blocks.
        assert config_file['data'] == default_cfg['data'], 'data block differs from default'
        assert config_file['features'] == default_cfg['features'], 'features block differs from default'

        print(f"\n=== {row['config_name']} case{case} -> {model_hash} ===")
        t0 = time.time()
        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'config_name': row['config_name'],
            'model_hash': model_hash,
            'held_out_case': case,
            'status': 'ok',
            'train_skill': None,
            'val_skill': None,
            'best_val_mse': None,
            'epochs_run': None,
            'error': None,
            'elapsed_s': None
        }

        try:
            res = train_one_fold(case, config_file, model_hash)
            train_skill, val_skill = sanity_gate(res)
            record['train_skill'] = train_skill
            record['val_skill'] = val_skill
            record['best_val_mse'] = res['best_val_mse']
            record['epochs_run'] = len(res['all_val_mse'])
        except Exception:
            record['status'] = 'failed'
            record['error'] = traceback.format_exc(limit=3).replace('\n', ' | ')
            print(f"FAILED {row['config_name']} case{case}")
            traceback.print_exc()

        record['elapsed_s'] = round(time.time() - t0, 1)
        pd.DataFrame([record]).to_csv(
            log_path, mode='a', header=not log_path.exists(), index=False)

        torch.cuda.empty_cache()