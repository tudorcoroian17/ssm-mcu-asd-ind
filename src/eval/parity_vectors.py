"""
Freezes a single-clip forward trace of a trained checkpoint.

Phase 4 hand-writes the SSM recurrence in C and needs a golden reference to
diff against, tensor by tensor (03_phase_2_ablation_and_loso.md handoff item 4,
05_phase_4_backbone_port.md step 4). This script produces it:
runs/case{N}/<hash>/parity_vectors.npz, plus a JSON sidecar recording exactly
what produced it.

Must run BEFORE _scan() and discretize() are refactored for streaming
execution. A trace generated afterward describes the refactored code rather
than the code that produced the reported results, which makes it useless as a
reference. Re-running it after the refactor and diffing the two files is the
cheapest available test that the refactor preserved the maths.

Loads existing checkpoints. Does not train.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT, load_config_by_name
from src.data.folds import get_fold
from src.features.baselines import apply_normalization
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.models.heads import PredictionHead

MANIFEST_PATH = PROJECT_ROOT / 'configs' / 'ablation' / '000_config_manifest.csv'

# A_bar and B_bar are (1, T, d_inner, d_state), about 5.6 MB each per block at
# the default config. Phase 4 diffs the recurrence at chosen steps, not at all
# 344, so keep a spread: the first few while the state is still filling, one
# mid-clip, and the last two, where accumulated drift is worst.
SELECTED_TIMESTEPS = (0, 1, 2, 3, 100, 171, 342, 343)
SUBSAMPLED_SUFFIXES = ('.A_bar', '.B_bar')
POOLING_MODES = ('mean', 'max', 'concat_mean_last')


class TensorCollector:
    """
    Exposes the same record(name, tensor) interface as RangeRecorder in
    checks/smoke/activation_ranges.py, so it drops into the existing
    range_recorder hook with no further model changes. Keeps whole tensors
    instead of min/max. A name recorded more than once per pass stacks along a
    new leading axis, which is how the hidden state arrives: _scan() records
    every tenth timestep.
    """

    def __init__(self):
        self.tensors = {}

    def record(self, name, tensor):
        self.tensors.setdefault(name, []).append(
            tensor.detach().to('cpu', dtype=torch.float32).numpy()
        )

    def to_dict(self):
        out = {}
        for name, chunks in self.tensors.items():
            arr = chunks[0] if len(chunks) == 1 else np.stack(chunks, axis=0)
            if name.endswith(SUBSAMPLED_SUFFIXES):
                arr = arr[:, list(SELECTED_TIMESTEPS)]
            out[name] = arr
        return out


def select_parity_clip(fold, clip_index=0):
    """
    Picks one validation-normal clip, deterministically.

    Sorts by file path rather than trusting DataFrame order: fold['val'] comes
    out of a seeded shuffle, so row 0 would move if val_split_seed ever changed.
    The path sort is stable across every config, which matters because Phase 4
    compares one clip against one stored trace.
    """
    val_normal = fold['val'][fold['val']['label'] == 'normal']
    ordered = val_normal.sort_values('path').reset_index(drop=True)
    return ordered.iloc[clip_index]


def dump_parity_vectors(cfg, held_out_case, model_hash, device='cpu', clip_index=0):
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / model_hash
    ckpt_path = base_dir / 'ckpt.pt'
    assert ckpt_path.exists(), f'no checkpoint at {ckpt_path}'

    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, _ = compute_normalization_stats(fold['train']['cache_path'], n_mels)

    row = select_parity_clip(fold, clip_index)
    log_mel = np.load(row['cache_path']).astype(np.float32)  # (T, n_mels), pre-normalization
    x = apply_normalization(log_mel, mean, std).astype(np.float32)

    model = SSMBackbone(**cfg['model']).to(device)
    head = PredictionHead(cfg['model']['d_model'], cfg['model']['n_mels']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    head.load_state_dict(ckpt['head'])
    model.eval()
    head.eval()

    collector = TensorCollector()
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, T, n_mels)

    with torch.no_grad():
        collector.record('model_input', x_t)
        seq_out = model(x_t, mode='sequence', range_recorder=collector)
        collector.record('head_output', head(seq_out))
        # mode='pooled' would re-run every block and double each recorded
        # tensor. forward() pools the same post-final_norm tensor, so pooling
        # seq_out directly is equivalent and keeps this to one pass.
        for pooling in POOLING_MODES:
            model.pooling = pooling
            collector.record(f'pooled_{pooling}', model._pool(seq_out))

    arrays = collector.to_dict()
    # Phase 3 validates the CMSIS-DSP log-mel against the unnormalized array;
    # Phase 4 validates the backbone against model_input. Both belong here,
    # along with the stats that connect them.
    arrays['log_mel_unnormalized'] = log_mel
    arrays['norm_mean'] = mean
    arrays['norm_std'] = std
    arrays['selected_timesteps'] = np.array(SELECTED_TIMESTEPS)

    out_path = base_dir / 'parity_vectors.npz'
    np.savez_compressed(out_path, **arrays)

    meta = {
        'source_wav': str(row['path']),
        'source_cache': str(row['cache_path']),
        'case_id': int(row['case_id']),
        'clip_index': clip_index,
        'held_out_case': held_out_case,
        'model_hash': model_hash,
        'seed': cfg['seed'],
        'model': cfg['model'],
        'features': cfg['features'],
        'training': cfg['training'],
        'ckpt_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'torch_version': torch.__version__,
        'subsampled_tensors': list(SUBSAMPLED_SUFFIXES),
        'selected_timesteps': list(SELECTED_TIMESTEPS),
        'tensors': {name: list(arr.shape) for name, arr in sorted(arrays.items())},
    }
    with open(base_dir / 'parity_vectors_meta.json', 'w') as f:
        json.dump(meta, f, indent=4)

    total_mb = sum(a.nbytes for a in arrays.values()) / 1024 ** 2
    compressed_mb = out_path.stat().st_size / 1024 ** 2
    print(f'  {len(arrays)} tensors, {total_mb:.1f} MB raw -> {compressed_mb:.1f} MB compressed')
    return arrays


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dump parity traces for every trained checkpoint on disk')
    parser.add_argument('--config-name', default=None,
                        help='restrict to one config file, e.g. 584eb86deba0.yaml')
    parser.add_argument('--outer', type=int, default=None,
                        help='restrict to one held-out case')
    parser.add_argument('--show-shapes', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest = pd.read_csv(MANIFEST_PATH)

    if args.config_name:
        manifest = manifest[manifest['config_name'] == args.config_name]
    if args.outer:
        manifest = manifest[manifest['held_out_case'] == args.outer]

    # Drive off what is actually on disk rather than off the filter: the same
    # config appears in the manifest at several horizon_k / target values, and
    # only the trained rows have a checkpoint.
    todo = [row for _, row in manifest.iterrows()
            if (PROJECT_ROOT / 'runs' / f"case{row['held_out_case']}"
                / row['model_hash'] / 'ckpt.pt').exists()]
    print(f'{len(todo)} trained checkpoints found')

    for row in todo:
        case = int(row['held_out_case'])
        print(f"\n=== {row['config_name']} case{case} -> {row['model_hash']} ===")
        cfg = load_config_by_name(row['config_name'])
        arrays = dump_parity_vectors(cfg, case, row['model_hash'], device=device)
        if args.show_shapes:
            for name, arr in sorted(arrays.items()):
                print(f'    {name:34s} {tuple(arr.shape)}')