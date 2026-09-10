"""
Offline activation int8 quantization parity check (Phase 5 rung 2 diagnostic).

Weights stay fp32 here -- this isolates the findings/150 question: does the
scan's internal arithmetic (delta, A_bar, B_bar, h) survive int8 on its own,
separated from any weight-quantization effect? This is a diagnostic ablation,
NOT a deployment scheme (activation-only quantizes nothing that ships smaller
or runs faster -- see the reasoning that motivated it).

Builds an ActivationQuantizer and passes it into the model's forward, which
quantize-dequantizes the selected activation group in place. Diffs the pooled
embedding against the fp32 model (quantizer=None) through the same forward.

Axes (see activation_quant.py):
  --group        scan | boundaries | all
  --granularity  per-tensor | per-channel
  --scale-source onthefly | ranges

Loads existing checkpoints, does not train.

Usage:
    python -m checks.smoke.activation_quant_parity \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        [--group scan|boundaries|all] [--granularity per-tensor|per-channel] \
        [--scale-source onthefly|ranges]
"""
import argparse
import json

import numpy as np
import torch

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.baselines import apply_normalization
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.eval.parity_vectors import select_parity_clip
from checks.smoke.quant.activation_quant import (
    ActivationQuantizer, ACTIVATION_GROUPS, GRANULARITIES, SCALE_SOURCES,
)


def load_ranges_scales(base_dir):
    """Build a name -> symmetric per-tensor scale dict from this run's
    ranges.json. Scale = max(|min|, |max|) / 127, the widest-value symmetric
    scale. findings/150 notes p99-based scales are worth trying for delta and
    A_bar; max-abs is the plain first choice, and the one a naive PTQ would
    pick."""
    with open(base_dir / 'ranges.json') as f:
        ranges = json.load(f)
    scales = {}
    for name, stats in ranges.items():
        max_abs = max(abs(stats['min']), abs(stats['max']))
        scales[name] = max_abs / 127.0 if max_abs > 0 else 1e-12
    return scales


def build_embeddings(cfg, held_out_case, model_hash, quantizer, device='cpu'):
    base_dir = PROJECT_ROOT / 'runs' / f'case{held_out_case}' / model_hash
    ckpt_path = base_dir / 'ckpt.pt'
    assert ckpt_path.exists(), f'no checkpoint at {ckpt_path}'

    fold = get_fold(held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, _ = compute_normalization_stats(fold['train']['cache_path'], n_mels)

    row = select_parity_clip(fold, clip_index=0)
    log_mel = np.load(row['cache_path']).astype(np.float32)
    x = apply_normalization(log_mel, mean, std).astype(np.float32)

    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        seq_out = model(x_t, mode='sequence', quantizer=quantizer)
        embeddings = {}
        for pooling in ('mean', 'max', 'concat_mean_last'):
            model.pooling = pooling
            embeddings[pooling] = model._pool(seq_out).squeeze(0).cpu().numpy()

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--held-out-case', type=int, required=True)
    parser.add_argument('--model-hash', required=True)
    parser.add_argument('--group', choices=ACTIVATION_GROUPS, default='scan')
    parser.add_argument('--granularity', choices=GRANULARITIES, default='per-tensor')
    parser.add_argument('--scale-source', choices=SCALE_SOURCES, default='onthefly')
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    base_dir = PROJECT_ROOT / 'runs' / f'case{args.held_out_case}' / args.model_hash

    scales = None
    if args.scale_source == 'ranges':
        scales = load_ranges_scales(base_dir)

    quantizer = ActivationQuantizer(
        group=args.group, granularity=args.granularity,
        scale_source=args.scale_source, scales=scales,
    )

    fp32 = build_embeddings(cfg, args.held_out_case, args.model_hash, quantizer=None)
    intq = build_embeddings(cfg, args.held_out_case, args.model_hash, quantizer=quantizer)

    print(f'\nActivation group: {args.group}   granularity: {args.granularity}   '
          f'scale: {args.scale_source}')
    print(f'quantized tensors ({len(quantizer.applied_names)}): '
          f'{", ".join(sorted(quantizer.applied_names))}')

    print('\nEmbedding parity, fp32 vs activation-int8:')
    print(f'  {"pooling":18s} {"max_abs_err":>12s} {"mean_abs_err":>12s}')
    for pooling in ('mean', 'max', 'concat_mean_last'):
        d = np.abs(intq[pooling] - fp32[pooling])
        print(f'  {pooling:18s} {d.max():12.6e} {d.mean():12.6e}')


if __name__ == '__main__':
    main()