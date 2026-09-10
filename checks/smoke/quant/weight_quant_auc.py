"""
Int8 weight-quantization AUC/pAUC impact, fp32 vs quantized, single case.

The Phase 5 rung-1 decision. weight_quant_parity.py shows how far the pooled
embedding drifts under a chosen scheme; this answers the question that matters
-- whether that drift moves the anomaly-detection AUC/pAUC, across all four
distance heads and three poolings.

Reuses src.eval unchanged: builds fp32 and int8 embeddings for one case, then
scores both with the identical DISTANCE_HEADS from auc_pauc.py and prints a
side-by-side delta. Never writes over the fp32 baseline artifacts.

The --weight-mode and --granularity axes match weight_quant_parity.py; their
definitions are imported from it so the two scripts cannot disagree about what
a scheme means. Loads existing checkpoints, does not train.

Usage:
    python -m checks.smoke.weight_quant_auc \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        [--weight-mode projections|all] [--granularity per-tensor|per-channel]
"""
import argparse

import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips
from src.models.backbone import SSMBackbone
from src.eval.embeddings import get_embeddings, POOLING_MODES
from src.eval.auc_pauc import DISTANCE_HEADS

from checks.smoke.quant.weight_quant_parity import (
    should_quantize, quantize_dequantize, WEIGHT_MODES, GRANULARITIES,
)


def load_model(cfg, base_dir, device, quantize,
               weight_mode='projections', granularity='per-tensor'):
    model = SSMBackbone(**cfg['model']).to(device)
    ckpt = torch.load(base_dir / 'ckpt.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if quantize:
        sd = model.state_dict()
        for name, w in sd.items():
            if should_quantize(name, weight_mode):
                deq, _ = quantize_dequantize(w, granularity)
                sd[name] = deq
        model.load_state_dict(sd)
    return model


def embeddings_for(model, fold, mean, std, device):
    """train + test embeddings per pooling mode -- the two splits the distance
    heads consume (train fits the head, test is scored)."""
    X_train = load_fold_clips(fold['train'], mean, std)
    X_test = load_fold_clips(fold['test'], mean, std)
    test_labels = (fold['test']['label'].values == 'anomaly').astype(int)

    out = {}
    for pooling in POOLING_MODES:
        model.pooling = pooling
        out[pooling] = (
            get_embeddings(model, X_train, device),
            get_embeddings(model, X_test, device),
        )
    return out, test_labels


def score_table(embs, test_labels, seed):
    """{(pooling, head): (auc, pauc)} over every pooling x head combination."""
    table = {}
    for pooling in POOLING_MODES:
        train_emb, test_emb = embs[pooling]
        for head_name, head_fn in DISTANCE_HEADS.items():
            scores, _ = head_fn(train_emb, test_emb, seed)
            auc = roc_auc_score(test_labels, scores)
            pauc = roc_auc_score(test_labels, scores, max_fpr=0.1)
            table[(pooling, head_name)] = (float(auc), float(pauc))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--held-out-case', type=int, required=True)
    parser.add_argument('--model-hash', required=True)
    parser.add_argument('--weight-mode', choices=WEIGHT_MODES, default='projections',
                        help='which weight tensors to quantize (see WEIGHT_MODE_SUFFIXES)')
    parser.add_argument('--granularity', choices=GRANULARITIES, default='per-tensor',
                        help='per-tensor (one scale) or per-channel (one scale per axis-0 channel)')
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = PROJECT_ROOT / 'runs' / f'case{args.held_out_case}' / args.model_hash

    fold = get_fold(args.held_out_case)
    n_mels = cfg['features']['n_mels']
    mean, std, _ = compute_normalization_stats(fold['train']['cache_path'], n_mels)

    print('building fp32 embeddings...')
    fp32_model = load_model(cfg, base_dir, device, quantize=False)
    fp32_embs, test_labels = embeddings_for(fp32_model, fold, mean, std, device)

    print(f'building weight-int8 embeddings [{args.weight_mode}/{args.granularity}]...')
    int8_model = load_model(cfg, base_dir, device, quantize=True,
                            weight_mode=args.weight_mode, granularity=args.granularity)
    int8_embs, _ = embeddings_for(int8_model, fold, mean, std, device)

    seed = cfg['seed']
    fp32_scores = score_table(fp32_embs, test_labels, seed)
    int8_scores = score_table(int8_embs, test_labels, seed)

    print(f'\ncase{args.held_out_case}  fp32 vs weight-int8 [{args.weight_mode}/{args.granularity}]  '
          f'(n_test={len(test_labels)}, anomalies={int(test_labels.sum())})')
    print(f'  {"pooling":18s} {"head":18s} '
          f'{"AUC f32":>8s} {"AUC i8":>8s} {"dAUC":>8s}   '
          f'{"pAUC f32":>9s} {"pAUC i8":>8s} {"dpAUC":>8s}')
    for pooling in POOLING_MODES:
        for head_name in DISTANCE_HEADS:
            a0, p0 = fp32_scores[(pooling, head_name)]
            a1, p1 = int8_scores[(pooling, head_name)]
            print(f'  {pooling:18s} {head_name:18s} '
                  f'{a0:8.4f} {a1:8.4f} {a1 - a0:+8.4f}   '
                  f'{p0:9.4f} {p1:8.4f} {p1 - p0:+8.4f}')


if __name__ == '__main__':
    main()