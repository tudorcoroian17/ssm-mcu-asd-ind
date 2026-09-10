"""
Offline int8 weight-quantization parity check.

Answers the Phase 5 rung-1 question -- does weight-only int8 survive the
scan? -- entirely in Python, before any C or export-script change. Loads a
trained checkpoint, quantizes a chosen set of weight tensors to int8 and
dequantizes them back to fp32, reloads the perturbed weights into the same
model, and diffs the pooled embedding against the unquantized fp32 model
through the identical forward pass src/eval/parity_vectors.py uses.

Two independent axes select what to run:

  --weight-mode  which tensors are quantized:
      projections  projection weight matrices + RMSNorm weights only; A, D,
                   and biases stay fp32 (the deliberate first rung).
      all          projections plus the recurrence params A and D and all
                   biases (the pass most likely to degrade -- see the note on
                   WEIGHT_MODE_SUFFIXES).

  --granularity  how each tensor is quantized:
      per-tensor   one scale per tensor. Cheapest; the first rung.
      per-channel  one scale per output channel (axis 0). More accurate, at
                   the cost of one fp32 scale per channel -- a flash cost this
                   script does not measure, only the accuracy side.

Weights only: activations stay fp32 here. Loads existing checkpoints, does
not train.

Usage:
    python -m checks.smoke.weight_quant_parity \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        [--weight-mode projections|all] [--granularity per-tensor|per-channel]
"""
import argparse

import numpy as np
import torch

from pathlib import Path

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.baselines import apply_normalization
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.eval.parity_vectors import select_parity_clip

# Weight tensors quantized under each mode. The mode name is what --weight-mode
# selects; suffixes are matched with `in`, so 'norms.' catches every per-block
# RMSNorm weight.
#
#   projections  learned weight matrices + norms only. A, D, and biases stay
#                fp32. The deliberate first rung (findings/460).
#   all          projections plus the recurrence params A and D and all biases.
#                A is precomputed -exp(A_log): strictly negative, wide dynamic
#                range, and the tensor findings/150 section 6 flags as most
#                sensitive. Expect this mode to be the one most likely to
#                degrade -- re-run decay_half_life.py against it, per
#                findings/150 recommendation 3.
WEIGHT_MODE_SUFFIXES = {
    'projections': (
        'in_proj.weight',
        'conv.weight',
        'x_proj.weight',
        'dt_proj.weight',
        'out_proj.weight',
        'norms.',
        'final_norm.weight',
    ),
}
WEIGHT_MODE_SUFFIXES['all'] = WEIGHT_MODE_SUFFIXES['projections'] + (
    'in_proj.bias',
    'conv.bias',
    'x_proj.bias',
    'dt_proj.bias',
    'out_proj.bias',
    '.A',      # matched with a leading dot to hit 'blocks.N.A', not substrings
    '.D',
)

WEIGHT_MODES = tuple(WEIGHT_MODE_SUFFIXES)
GRANULARITIES = ('per-tensor', 'per-channel')


def should_quantize(param_name, mode='projections'):
    return any(s in param_name for s in WEIGHT_MODE_SUFFIXES[mode])


def quantize_dequantize(w, granularity='per-tensor'):
    """Symmetric int8 quantize-dequantize. Returns (dequantized fp32 tensor, stats).

    per-tensor:   one scale for the whole tensor.
    per-channel:  one scale per output channel (axis 0). Falls back to
                  per-tensor for tensors with fewer than two dims, where
                  per-channel would be per-element -- one fp32 scale per int8
                  value, larger than fp32 and pointless (norm weights, D,
                  biases).
    """
    w_np = w.detach().cpu().numpy().astype(np.float32)
    overall_max = float(np.max(np.abs(w_np)))
    if overall_max == 0.0:
        return w.clone(), {'max_abs': 0.0, 'max_abs_err': 0.0, 'snr_db': float('inf')}

    if granularity == 'per-channel' and w_np.ndim >= 2:
        reduce_axes = tuple(range(1, w_np.ndim))
        max_abs = np.max(np.abs(w_np), axis=reduce_axes, keepdims=True)
        scale = np.where(max_abs == 0.0, 1.0, max_abs / 127.0)
    else:
        scale = overall_max / 127.0

    q = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scale).astype(np.float32)

    err = np.abs(deq - w_np)
    signal = np.mean(w_np ** 2)
    noise = np.mean((deq - w_np) ** 2)
    snr_db = 10.0 * np.log10(signal / noise) if noise > 0 else float('inf')
    stats = {
        'max_abs': overall_max,
        'max_abs_err': float(np.max(err)),
        'snr_db': float(snr_db),
    }
    return torch.from_numpy(deq).reshape(w.shape).to(w.dtype), stats


def build_embeddings(cfg, held_out_case, model_hash, quantize,
                     weight_mode='projections', granularity='per-tensor', device='cpu',
                     streaming=False):
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

    per_tensor_stats = {}
    if quantize:
        sd = model.state_dict()
        for name, w in sd.items():
            if should_quantize(name, weight_mode):
                deq, stats = quantize_dequantize(w, granularity)
                sd[name] = deq
                per_tensor_stats[name] = stats
        model.load_state_dict(sd)

    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        seq_out = model(x_t, mode='sequence', streaming=streaming)
        embeddings = {}
        for pooling in ('mean', 'max', 'concat_mean_last'):
            model.pooling = pooling
            embeddings[pooling] = model._pool(seq_out).squeeze(0).cpu().numpy()

    return embeddings, per_tensor_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--held-out-case', type=int, required=True)
    parser.add_argument('--model-hash', required=True)
    parser.add_argument('--weight-mode', choices=WEIGHT_MODES, default='projections',
                        help='which weight tensors to quantize (see WEIGHT_MODE_SUFFIXES)')
    parser.add_argument('--granularity', choices=GRANULARITIES, default='per-tensor',
                        help='per-tensor (one scale) or per-channel (one scale per axis-0 channel)')
    parser.add_argument('--dump-reference-dir', default=None,
                        help="if set, save this run's quantized mean-pooling "
                             "embedding to <dir>/reference_embedding.npy, for "
                             "mcu/check_backbone_parity.py to compare against "
                             "(matches ssm_backbone.c, which only computes "
                             "mean pooling)")
    parser.add_argument('--streaming', action='store_true',
                        help='use _scan_streaming instead of the batched path -- '
                             'matches ssm_backbone.c\'s evaluation order. Use this '
                             'when generating a reference for --dump-reference-dir.')
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)

    fp32, _ = build_embeddings(cfg, args.held_out_case, args.model_hash, quantize=False,
                               weight_mode=args.weight_mode, granularity=args.granularity,
                               streaming=args.streaming)
    int8, stats = build_embeddings(cfg, args.held_out_case, args.model_hash, quantize=True,
                                   weight_mode=args.weight_mode, granularity=args.granularity,
                                   streaming=args.streaming)

    print(f'\nWeight mode: {args.weight_mode}   granularity: {args.granularity}')
    print('\nPer-tensor weight quantization error (worst SNR first):')
    print(f'  {"tensor":36s} {"max|w|":>10s} {"max_err":>10s} {"SNR dB":>8s}')
    for name in sorted(stats, key=lambda n: stats[n]['snr_db']):
        s = stats[name]
        print(f'  {name:36s} {s["max_abs"]:10.4f} {s["max_abs_err"]:10.6f} {s["snr_db"]:8.1f}')

    print('\nEmbedding parity, fp32 vs weight-int8:')
    print(f'  {"pooling":18s} {"max_abs_err":>12s} {"mean_abs_err":>12s}')
    for pooling in ('mean', 'max', 'concat_mean_last'):
        d = np.abs(int8[pooling] - fp32[pooling])
        print(f'  {pooling:18s} {d.max():12.6e} {d.mean():12.6e}')

    if args.dump_reference_dir:
        out_path = Path(args.dump_reference_dir) / 'reference_embedding.npy'
        np.save(out_path, int8['mean'])
        print(f'\nWrote quantized mean-pooling reference to {out_path}')


if __name__ == '__main__':
    main()