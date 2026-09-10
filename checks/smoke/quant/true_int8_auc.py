"""
True int8 (real int32-accumulate arithmetic, not fake-quant) AUC/pAUC
impact, fp32 vs quantized, single case. The decisive check for
true_int8_sim.py's embedding-distance results -- findings/520/521 already
showed embedding distance and AUC don't reliably track each other, so a
large distance figure alone (see the deployment-matrix conversation this
was built in) isn't a verdict on its own.

Calibration and weight/LUT preparation happen ONCE (prepare_quantized_model),
reused across every clip -- this is what makes a full train+test sweep
tractable. Expect several minutes to run, not seconds: this simulator does
genuinely more per-clip work than any fake-quant check in this project's
history (dozens of quantize points per frame, not a handful).

Usage:
    python -m checks.smoke.quant.true_int8_auc \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --h-width int8
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.baselines import apply_normalization
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.eval.auc_pauc import DISTANCE_HEADS

from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import (
    H_WIDTHS, calibrate_missing_ranges, prepare_quantized_model, quantized_forward,
)


def embeddings_for_split(rows, mean, std, sd, weights, scales, luts_per_layer, ranges,
                         d_model, d_inner, d_state, d_conv, n_layers, h_width, label):
    out = np.zeros((len(rows), d_model))
    t0 = time.time()
    for idx, (_, row) in enumerate(rows.iterrows()):
        log_mel = np.load(row["cache_path"]).astype(np.float32)
        x = apply_normalization(log_mel, mean, std).astype(np.float32)
        out[idx] = quantized_forward(x, sd, weights, scales, luts_per_layer, ranges,
                                     d_model, d_inner, d_state, d_conv, n_layers, h_width)
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f'  {label}: {idx + 1}/{len(rows)} clips '
                  f'({elapsed:.0f}s elapsed, ~{elapsed / (idx + 1) * len(rows):.0f}s total)')
    return out


def score_table(train_emb, test_emb, test_labels, seed):
    table = {}
    for head_name, head_fn in DISTANCE_HEADS.items():
        scores, _ = head_fn(train_emb, test_emb, seed)
        table[head_name] = (roc_auc_score(test_labels, scores),
                            roc_auc_score(test_labels, scores, max_fpr=0.1))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--h-width", choices=list(H_WIDTHS), default="int8")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dt_rank = max(1, d_model // 16)

    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    model = SSMBackbone(**m)
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    fold = get_fold(args.held_out_case)
    mean, std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    ranges = load_ranges_scales(base_dir)

    print("Calibrating missing ranges (dt_proj/conv/norm hooks)...")
    hook_scales = calibrate_missing_ranges(model, fold, mean, std, n_layers, dt_rank)

    print("Preparing quantized weights and LUTs (once, reused across all clips)...")
    weights, scales, luts_per_layer = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, args.h_width)

    print("Building fp32 embeddings (existing, fast path)...")
    from src.features.baselines import load_fold_clips
    from src.eval.embeddings import get_embeddings
    X_train = load_fold_clips(fold["train"], mean, std)
    X_test = load_fold_clips(fold["test"], mean, std)
    test_labels = (fold["test"]["label"].values == "anomaly").astype(int)
    model.pooling = "mean"
    fp32_train = get_embeddings(model, X_train, "cpu")
    fp32_test = get_embeddings(model, X_test, "cpu")

    print(f"Building true-int8 [{args.h_width}] embeddings -- this is the slow part...")
    q_train = embeddings_for_split(fold["train"], mean, std, sd, weights, scales,
                                   luts_per_layer, ranges, d_model, d_inner, d_state,
                                   d_conv, n_layers, args.h_width, "train")
    q_test = embeddings_for_split(fold["test"], mean, std, sd, weights, scales,
                                  luts_per_layer, ranges, d_model, d_inner, d_state,
                                  d_conv, n_layers, args.h_width, "test")

    seed = cfg["seed"]
    fp32_scores = score_table(fp32_train, fp32_test, test_labels, seed)
    q_scores = score_table(q_train, q_test, test_labels, seed)

    print(f"\ncase{args.held_out_case}  fp32 vs true-int8 [{args.h_width}]  "
          f"(n_test={len(test_labels)}, anomalies={int(test_labels.sum())})")
    print(f'  {"head":18s} {"AUC f32":>8s} {"AUC iq":>8s} {"dAUC":>8s}   '
          f'{"pAUC f32":>9s} {"pAUC iq":>8s} {"dpAUC":>8s}')
    for head_name in DISTANCE_HEADS:
        a0, p0 = fp32_scores[head_name]
        a1, p1 = q_scores[head_name]
        print(f'  {head_name:18s} {a0:8.4f} {a1:8.4f} {a1 - a0:+8.4f}   '
              f'{p0:9.4f} {p1:8.4f} {p1 - p0:+8.4f}')


if __name__ == "__main__":
    main()