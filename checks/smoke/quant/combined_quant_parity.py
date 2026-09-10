"""
Combined weight + activation int8 parity check and reference dump.

Neither weight_quant_parity.py nor activation_quant_parity.py quantizes
both at once -- each leaves the other axis fp32. This is the first check
of the combination, and the only source of a valid on-device reference for
any weight+activation deployment scheme (06_phase5_nucleo_deployment_matrix.md
items 4-6). Reuses both quantizers unchanged: quantize_dequantize (weights)
and ActivationQuantizer (activations), applied to the same model in the
same forward pass, so nothing here reimplements either axis's math.

Uses --streaming by default (unlike the two single-axis scripts, which
default to batched) since the whole point of this script is producing a
device-comparable reference -- see findings/530 for why streaming vs.
batched evaluation order was ruled out as a discrepancy source, and why
comparing device output against a batched-path reference wasted real
debugging time before that was confirmed.

Loads existing checkpoints, does not train.

Usage:
    python -m checks.smoke.quant.combined_quant_parity \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --weight-mode all --granularity per-tensor \
        --activation-group boundaries \
        --dump-reference-dir mcu/deploy/case1/weight_act_boundaries_ranges
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.baselines import apply_normalization
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.eval.parity_vectors import select_parity_clip

from checks.smoke.quant.weight_quant_parity import (
    should_quantize, quantize_dequantize, WEIGHT_MODES, GRANULARITIES,
)
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.activation_quant import ActivationQuantizer, ACTIVATION_GROUPS


def build_embedding(cfg, held_out_case, model_hash, weight_mode, granularity,
                    activation_group, quantize, device='cpu', streaming=True):
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

    quantizer = None
    if quantize:
        sd = model.state_dict()
        for name, w in sd.items():
            if should_quantize(name, weight_mode):
                deq, _ = quantize_dequantize(w, granularity)
                sd[name] = deq
        model.load_state_dict(sd)

        if activation_group != 'none':
            scales = load_ranges_scales(base_dir)
            quantizer = ActivationQuantizer(
                group=activation_group, granularity='per-tensor',
                scale_source='ranges', scales=scales,
            )

    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        seq_out = model(x_t, mode='sequence', streaming=streaming, quantizer=quantizer)
        model.pooling = 'mean'
        embedding = model._pool(seq_out).squeeze(0).cpu().numpy()

    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--held-out-case', type=int, required=True)
    parser.add_argument('--model-hash', required=True)
    parser.add_argument('--weight-mode', choices=WEIGHT_MODES, default='all')
    parser.add_argument('--granularity', choices=GRANULARITIES, default='per-tensor')
    parser.add_argument('--activation-group', choices=ACTIVATION_GROUPS, default='boundaries')
    parser.add_argument('--dump-reference-dir', required=True,
                        help='mcu/deploy/case<N>/<scheme>/ -- writes reference_embedding.npy there')
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)

    fp32 = build_embedding(cfg, args.held_out_case, args.model_hash,
                           args.weight_mode, args.granularity, args.activation_group,
                           quantize=False)
    quantized = build_embedding(cfg, args.held_out_case, args.model_hash,
                                args.weight_mode, args.granularity, args.activation_group,
                                quantize=True)

    d = np.abs(quantized - fp32)
    print(f'\nWeight mode: {args.weight_mode}/{args.granularity}   '
          f'Activation group: {args.activation_group}')
    print(f'Embedding parity, fp32 vs combined-int8 (mean pooling only -- matches device):')
    print(f'  max_abs_err: {d.max():.6e}   mean_abs_err: {d.mean():.6e}')

    out_path = Path(args.dump_reference_dir) / 'reference_embedding.npy'
    np.save(out_path, quantized)
    print(f'\nWrote combined quantized mean-pooling reference to {out_path}')


if __name__ == '__main__':
    main()