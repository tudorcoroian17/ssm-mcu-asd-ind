"""
Activation int8 quantization AUC/pAUC impact, fp32 vs quantized, single case.

The rung-2 diagnostic decision. activation_quant_parity.py shows how far the
embedding drifts under a chosen activation-quant scheme; this answers whether
that drift moves the anomaly-detection AUC/pAUC across all four distance heads
and three poolings. Weights stay fp32 -- this isolates the activation effect.
Diagnostic ablation, not a deployment scheme.

Reuses src.eval scoring unchanged, and imports score_table from the weight AUC
driver so the two cannot disagree about how AUC is computed. Builds fp32 and
activation-int8 embeddings for one case; never writes over fp32 artifacts.

Axes match activation_quant_parity.py. Loads checkpoints, does not train.

Usage:
    python -m checks.smoke.activation_quant_auc \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        [--group scan|boundaries|all] [--granularity per-tensor|per-channel] \
        [--scale-source onthefly|ranges]
"""
import argparse

import numpy as np
import torch

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from src.eval.embeddings import POOLING_MODES

from checks.smoke.quant.activation_quant import (
    ActivationQuantizer, ACTIVATION_GROUPS, GRANULARITIES, SCALE_SOURCES,
)
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.weight_quant_auc import score_table  # pure: embeddings -> AUC table


def get_embeddings_q(model, X, device, quantizer, batch_size=128):
    """Mirror of src.eval.embeddings.get_embeddings, but threads the quantizer
    into the forward. Kept local so src/eval/embeddings.py stays untouched."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            emb = model(batch, mode='pooled', quantizer=quantizer)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


def load_model(cfg, base_dir, device):
    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def embeddings_for(model, fold, mean, std, device, quantizer):
    X_train = load_fold_clips(fold['train'], mean, std)
    X_test = load_fold_clips(fold['test'], mean, std)
    test_labels = (fold['test']['label'].values == 'anomaly').astype(int)

    out = {}
    for pooling in POOLING_MODES:
        model.pooling = pooling
        out[pooling] = (
            get_embeddings_q(model, X_train, device, quantizer),
            get_embeddings_q(model, X_test, device, quantizer),
        )
    return out, test_labels


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = PROJECT_ROOT / 'runs' / f'case{args.held_out_case}' / args.model_hash

    fold = get_fold(args.held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, _ = compute_normalization_stats(fold['train']['cache_path'], n_mels)

    scales = load_ranges_scales(base_dir) if args.scale_source == 'ranges' else None

    model = load_model(cfg, base_dir, device)

    print('building fp32 embeddings...')
    fp32_embs, test_labels = embeddings_for(model, fold, mean, std, device, quantizer=None)

    quantizer = ActivationQuantizer(
        group=args.group, granularity=args.granularity,
        scale_source=args.scale_source, scales=scales,
    )
    print(f'building activation-int8 embeddings [{args.group}/{args.granularity}/{args.scale_source}]...')
    intq_embs, _ = embeddings_for(model, fold, mean, std, device, quantizer=quantizer)

    seed = cfg['seed']
    fp32_scores = score_table(fp32_embs, test_labels, seed)
    intq_scores = score_table(intq_embs, test_labels, seed)

    print(f'\ncase{args.held_out_case}  fp32 vs activation-int8 '
          f'[{args.group}/{args.granularity}/{args.scale_source}]  '
          f'(n_test={len(test_labels)}, anomalies={int(test_labels.sum())})')
    print(f'quantized tensors ({len(quantizer.applied_names)}): '
          f'{", ".join(sorted(quantizer.applied_names))}')
    print(f'  {"pooling":18s} {"head":18s} '
          f'{"AUC f32":>8s} {"AUC iq":>8s} {"dAUC":>8s}   '
          f'{"pAUC f32":>9s} {"pAUC iq":>8s} {"dpAUC":>8s}')
    for pooling in POOLING_MODES:
        for head_name in ('euclidean', 'mahalanobis', 'knn_full', 'knn_clustered_16'):
            a0, p0 = fp32_scores[(pooling, head_name)]
            a1, p1 = intq_scores[(pooling, head_name)]
            print(f'  {pooling:18s} {head_name:18s} '
                  f'{a0:8.4f} {a1:8.4f} {a1 - a0:+8.4f}   '
                  f'{p0:9.4f} {p1:8.4f} {p1 - p0:+8.4f}')


if __name__ == '__main__':
    main()