"""
Per-tensor activation range calibration for Phase 5 quantization.

Loads a trained checkpoint and runs one instrumented forward pass over the
fold's own validation-normal clips, recording exact min/max and approximate
percentiles for every named activation tensor along the SSM block's
compute path. Output: runs/case{N}/<hash>/ranges.json.

Validation-normal, not train or test: same choice as the threshold methods
(01_eval_spec.md section 9) -- data the model never trained on, but from
the same machine population, which is the right calibration set for a
deployment quantizer. Held-out case stays untouched, as everywhere else.

Does NOT retrain. Loads an existing checkpoint.
"""
import json

import numpy as np
import torch

from runs.compute_hash import train_config_hash
from src.config import load_config, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from src.models.heads import PredictionHead

class RangeRecorder:
    """
    Exact global min/max (O(1) memory) plus a bounded random sample per
    named tensor, accumulated across an entire calibration pass. Min/max
    is exact so Phase 5 knows the true worst case; percentiles come from
    the sample so one outlier clip can't dictate the working scale.
    """
    SAMPLES_PER_BATCH = 2000

    def __init__(self):
        self.min = {}
        self.max = {}
        self.samples = {}

    def record(self, name, tensor):
        flat = tensor.detach().reshape(-1)
        t_min = flat.min().item()
        t_max = flat.max().item()
        self.min[name] = min(self.min.get(name, t_min), t_min)
        self.max[name] = max(self.max.get(name, t_max), t_max)

        n = flat.numel()
        k = min(self.SAMPLES_PER_BATCH, n)
        idx = torch.randperm(n, device=flat.device)[:k]
        self.samples.setdefault(name, []).append(flat[idx].cpu())

    def to_dict(self, percentiles=(0.1, 1, 50, 99, 99.9)):
        out = {}
        for name in sorted(self.min):
            all_samples = torch.cat(self.samples[name]).numpy()
            pcts = np.percentile(all_samples, percentiles)
            out[name] = {
                'min': self.min[name],
                'max': self.max[name],
                **{f'p{p}': float(v) for p, v in zip(percentiles, pcts)},
                'n_samples': int(all_samples.size),
            }
        return out

def run_case(held_out_case, cfg, device):
    dir_name = train_config_hash(cfg, held_out_case)
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / dir_name

    model = SSMBackbone(**cfg['model']).to(device)
    head = PredictionHead(cfg['model']['d_model'], cfg['model']['n_mels']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    head.load_state_dict(ckpt['head'])
    model.eval()
    head.eval()

    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, _ = compute_normalization_stats(fold['train']['cache_path'], n_mels)
    val_normal_rows = fold['val'][fold['val']['label'] == 'normal']
    X_val = load_fold_clips(val_normal_rows, mean, std)

    recorder = RangeRecorder()
    batch_size = cfg['training']['batch_size']

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            x = torch.from_numpy(X_val[i:i + batch_size]).float().to(device)
            recorder.record('model_input', x)
            seq_out = model(x, mode='sequence', range_recorder=recorder)
            pred = head(seq_out)
            recorder.record('head_output', pred)

    ranges = recorder.to_dict()
    out_path = base_dir / 'ranges.json'
    with open(out_path, 'w') as f:
        json.dump(ranges, f, indent=4)

    print(f'case{held_out_case}: {len(ranges)} tensors -> {out_path}')
    for name, stats in ranges.items():
        print(f"  {name:30s} min={stats['min']:10.4f}  p1={stats['p1']:10.4f}  "
              f"p50={stats['p50']:10.4f}  p99={stats['p99']:10.4f}  max={stats['max']:10.4f}")


if __name__ == '__main__':
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for case in [1, 2, 3, 4]:
        print(f'\n=== held_out_case {case} ===')
        run_case(case, cfg, device)