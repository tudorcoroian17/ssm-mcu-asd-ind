"""
-----------------------------------------------------------------------------
**ARCHIVED** - Use the export_deploy_matrix.py script to generate full deploy
packages. This version is no longer supported or tested.
-----------------------------------------------------------------------------
Export reference embeddings (centroid + 16-cluster KMeans centers) and an
on-device decision threshold for the euclidean and knn_clustered_16 heads,
for either quantization scheme this project can build full-dataset
embeddings under:

    --scheme combined    weight int8 + activation-boundaries int8 (the
                          scheme already flashed as weight_act_boundaries_
                          ranges) -- mirrors combined_quant_parity.py's
                          single-clip build_embedding(), generalized to a
                          full split.
    --scheme true_int8   real int32-accumulate arithmetic
                          (checks/smoke/quant/true_int8_sim.py) at a chosen
                          --h-width. Reuses true_int8_auc.py's per-clip
                          embedding function unchanged.

WHY RECOMPUTE train_emb/calib_normal HERE INSTEAD OF REUSING CACHED fp32
EMBEDDINGS: the device produces quantized embeddings, not fp32 ones. Baking
in an fp32-derived centroid/cluster set would compare the device's
quantized query against a reference built in a different numeric domain --
exactly the mismatch flagged when this deployment was first scoped. Every
reference vector here is built from THIS scheme's own quantized forward
pass, matching what the device itself produces.

THRESHOLD: parametric fitting by default (src/eval/thresholds.py::parametric_threshold),
fit on THIS scheme's calib_normal scores (fold['calib_normal'] -- same
held-out machine, disjoint from train and from the balanced test split).
This is findings/140 sec 9's actual deployment recommendation for
knn_full/knn_clustered_16 -- NOT plain EVT, which that document's own sec
6-7 show becoming unreliable at aggressive false-alarm targets (it fits
only the top ~10% tail of calib_normal, ~108 points here; parametric uses
the full ~1085-point sample via MLE + AIC family selection). EVT remains
available via --threshold-method evt for direct comparison. Default target
false-alarm rate is 0.01 (p99_equiv, the middle of thresholds.py's own
FAR_BUDGETS) -- override with --target-far if a different rate is wanted;
nothing here defends 1% as uniquely correct.

OUTPUT: writes ssm_head_ref.h/.c into --out-dir, copies the canonical
scheme-agnostic ssm_distance_head.h/.c alongside them (same
copy-the-canonical-source convention export_ssm_weights.py uses for
ssm_backbone.h/.c), and writes head_export_diagnostics.json (AUC/pAUC and
threshold-based precision/recall/accuracy/F1 on the test split) for
comparison against whatever the device reports once flashed.

Usage (combined / boundaries scheme):
    python mcu/export_reference_heads.py \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --scheme combined --weight-mode all --granularity per-tensor \
        --activation-group boundaries \
        --out-dir mcu/deploy/case1/weight_act_boundaries_ranges

Usage (true int8, run once per h-width):
    python mcu/export_reference_heads.py \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --scheme true_int8 --h-width int8 \
        --out-dir mcu/deploy/case1/true_int8_h8
    python mcu/export_reference_heads.py \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --scheme true_int8 --h-width int16 \
        --out-dir mcu/deploy/case1/true_int8_h16
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips, apply_normalization
from src.models.backbone import SSMBackbone
from src.eval.auc_pauc import DISTANCE_HEADS
from src.eval.thresholds import evt_threshold, parametric_threshold, secondary_metrics

from checks.smoke.quant.weight_quant_parity import (
    should_quantize, quantize_dequantize, WEIGHT_MODES, GRANULARITIES,
)
from checks.smoke.quant.activation_quant import ActivationQuantizer, ACTIVATION_GROUPS
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import (
    H_WIDTHS, calibrate_missing_ranges, prepare_quantized_model, quantized_forward,
)

# Canonical, hand-maintained distance-head sources -- never generated.
# Copied byte-for-byte into every --out-dir, same convention as
# export_ssm_weights.py's BACKBONE_SRC_DIR. Two directories, IDENTICAL
# filenames and public API in both (SSMHeadResult, SSMDistanceHead_Score) --
# same "same filename, incompatible internals, never linked together"
# convention ssm_backbone.h/.c already use across ssm_backbone_src/ vs
# ssm_true_int8_src/. This is what lets a single main.c work for every
# scheme without scheme-specific branching.
HEAD_SRC_DIR_FLOAT = Path(__file__).parent / "ssm_head_src"
HEAD_SRC_DIR_TRUE_INT8 = Path(__file__).parent / "ssm_head_src_true_int8"
HEAD_FILES = ("ssm_distance_head.h", "ssm_distance_head.c")


def format_c_float(v):
    s = f"{v:.9g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s + "f"


def c_array_1d(name, arr):
    flat = np.asarray(arr, dtype=np.float32).flatten()
    values = ", ".join(format_c_float(v) for v in flat)
    return f"const float {name}[{flat.size}] = {{ {values} }};"


def c_array_2d(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    n_rows, n_cols = arr.shape
    rows = ",\n    ".join(
        "{ " + ", ".join(format_c_float(v) for v in row) + " }" for row in arr
    )
    return f"const float {name}[{n_rows}][{n_cols}] = {{\n    {rows}\n}};"


def c_int8_array_1d(name, arr):
    flat = np.asarray(arr, dtype=np.int8).flatten()
    values = ", ".join(str(int(v)) for v in flat)
    return f"const int8_t {name}[{flat.size}] = {{ {values} }};"


def c_int8_array_2d(name, arr):
    arr = np.asarray(arr, dtype=np.int8)
    n_rows, n_cols = arr.shape
    rows = ",\n    ".join("{ " + ", ".join(str(int(v)) for v in row) + " }" for row in arr)
    return f"const int8_t {name}[{n_rows}][{n_cols}] = {{\n    {rows}\n}};"


def quantize_ref_int8(real_arr, scale, label):
    """Quantizes a reference vector (centroid or a cluster center) to int8
    against a GIVEN scale (ssm_final_norm_scale -- the same scale the
    embedding itself lives on, never independently re-derived from this
    array's own range). Warns, rather than silently clipping without
    comment, if any entry would have exceeded the int8 range before
    clipping -- expected to be rare (a centroid/cluster center is a mean
    over many already-bounded embeddings, so its magnitude is usually
    smaller than any single embedding's), but worth surfacing if it ever
    happens rather than treating it as routine."""
    scaled = np.asarray(real_arr, dtype=np.float64) / scale
    n_clipped = int(np.sum(np.abs(scaled) > 127.0))
    if n_clipped > 0:
        print(f"  WARNING: {label} had {n_clipped} entr{'y' if n_clipped == 1 else 'ies'} "
              f"clip at int8 export -- centroid/cluster magnitude exceeded the "
              f"embedding's own int8 range at ssm_final_norm_scale.")
    q = np.clip(np.round(scaled), -127, 127).astype(np.int8)
    return q


# ---------------------------------------------------------------------------
# 'combined' scheme: weight int8 + activation-boundaries int8, exactly what
# checks/smoke/quant/combined_quant_parity.py builds for a single clip --
# generalized here to a whole split.
# ---------------------------------------------------------------------------

def build_combined_model(cfg, base_dir, device, weight_mode, granularity, activation_group):
    model = SSMBackbone(**cfg["model"]).to(device)
    ckpt = torch.load(base_dir / "ckpt.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sd = model.state_dict()
    for name, w in sd.items():
        if should_quantize(name, weight_mode):
            deq, _ = quantize_dequantize(w, granularity)
            sd[name] = deq
    model.load_state_dict(sd)

    quantizer = None
    if activation_group != "none":
        scales = load_ranges_scales(base_dir)
        quantizer = ActivationQuantizer(
            group=activation_group, granularity="per-tensor",
            scale_source="ranges", scales=scales,
        )
    return model, quantizer


def combined_embeddings(model, quantizer, X, device, batch_size=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            emb = model(batch, mode="pooled", quantizer=quantizer)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 'true_int8' scheme: real int32-accumulate arithmetic, per-clip loop,
# reusing true_int8_sim.py's per-clip function unchanged.
# ---------------------------------------------------------------------------

def true_int8_embeddings(rows, mean, std, sd, weights, scales, luts_per_layer, ranges,
                         d_model, d_inner, d_state, d_conv, n_layers, h_width, label):
    out = np.zeros((len(rows), d_model))
    for idx, (_, row) in enumerate(rows.iterrows()):
        log_mel = np.load(row["cache_path"]).astype(np.float32)
        x = apply_normalization(log_mel, mean, std).astype(np.float32)
        out[idx] = quantized_forward(x, sd, weights, scales, luts_per_layer, ranges,
                                     d_model, d_inner, d_state, d_conv, n_layers, h_width)
        if (idx + 1) % 25 == 0:
            print(f"  {label}: {idx + 1}/{len(rows)} clips")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scheme", choices=["combined", "true_int8"], required=True)
    parser.add_argument("--weight-mode", choices=WEIGHT_MODES, default="all",
                        help="combined scheme only")
    parser.add_argument("--granularity", choices=GRANULARITIES, default="per-tensor",
                        help="combined scheme only")
    parser.add_argument("--activation-group", choices=ACTIVATION_GROUPS, default="boundaries",
                        help="combined scheme only")
    parser.add_argument("--h-width", choices=list(H_WIDTHS), default="int8",
                        help="true_int8 scheme only")
    parser.add_argument("--target-far", type=float, default=0.01,
                        help="EVT target false-alarm rate; 0.01 = p99_equiv "
                             "(see thresholds.py FAR_BUDGETS). Not a settled "
                             "choice -- override if a different rate is wanted.")
    parser.add_argument("--threshold-method", choices=["parametric", "evt"], default="parametric",
                        help="findings/140 sec 9 recommends parametric_p99_equiv over "
                             "plain EVT/percentile for knn_full/knn_clustered_16 -- EVT "
                             "fits only the top ~10%% tail of calib_normal (~108 points "
                             "here) and that document's own sec 6-7 document real "
                             "collapse risk at aggressive quantiles from a tail sample "
                             "that small. parametric uses the full calib_normal sample "
                             "(~1085 points) via MLE + AIC family selection instead. "
                             "'evt' kept available for direct comparison, not because "
                             "it's the recommended default.")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    device = "cpu"
    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash

    fold = get_fold(args.held_out_case)
    n_mels = cfg["features"]["n_mels"]
    mean, std, _ = compute_normalization_stats(fold["train"]["cache_path"], n_mels)
    test_labels = (fold["test"]["label"].values == "anomaly").astype(int)

    if args.scheme == "combined":
        model, quantizer = build_combined_model(
            cfg, base_dir, device, args.weight_mode, args.granularity, args.activation_group)
        model.pooling = "mean"
        X_train = load_fold_clips(fold["train"], mean, std)
        X_test = load_fold_clips(fold["test"], mean, std)
        X_calib = load_fold_clips(fold["calib_normal"], mean, std)
        print(f"Building 'combined' [{args.weight_mode}/{args.granularity}/"
              f"{args.activation_group}] embeddings for train/test/calib_normal...")
        train_emb = combined_embeddings(model, quantizer, X_train, device)
        test_emb = combined_embeddings(model, quantizer, X_test, device)
        calib_emb = combined_embeddings(model, quantizer, X_calib, device)
        scheme_desc = f"combined/{args.weight_mode}/{args.granularity}/{args.activation_group}"

    else:  # true_int8
        d_model, d_state = m["d_model"], m["d_state"]
        d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
        dt_rank = max(1, d_model // 16)

        model = SSMBackbone(**m)
        ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        sd = model.state_dict()

        ranges = load_ranges_scales(base_dir)
        print("Calibrating missing ranges (dt_proj/conv/norm hooks)...")
        hook_scales = calibrate_missing_ranges(model, fold, mean, std, n_layers, dt_rank)
        print("Preparing quantized weights and LUTs (once, reused across all clips)...")
        weights, scales, luts_per_layer = prepare_quantized_model(
            sd, ranges, hook_scales, d_inner, n_layers, args.h_width)

        print(f"Building 'true_int8' [h={args.h_width}] embeddings for "
              f"train/test/calib_normal -- this is the slow part...")
        train_emb = true_int8_embeddings(fold["train"], mean, std, sd, weights, scales,
                                         luts_per_layer, ranges, d_model, d_inner, d_state,
                                         d_conv, n_layers, args.h_width, "train")
        test_emb = true_int8_embeddings(fold["test"], mean, std, sd, weights, scales,
                                        luts_per_layer, ranges, d_model, d_inner, d_state,
                                        d_conv, n_layers, args.h_width, "test")
        calib_emb = true_int8_embeddings(fold["calib_normal"], mean, std, sd, weights, scales,
                                         luts_per_layer, ranges, d_model, d_inner, d_state,
                                         d_conv, n_layers, args.h_width, "calib_normal")
        scheme_desc = f"true_int8/h={args.h_width}"

    seed = cfg["seed"]

    # euclidean: centroid straight from DISTANCE_HEADS, so this can never
    # disagree with how auc_pauc.py itself defines the head.
    euclidean_scores, euclidean_extra = DISTANCE_HEADS["euclidean"](train_emb, test_emb, seed)
    centroid = euclidean_extra["centroid"]

    # knn_clustered_16: DISTANCE_HEADS only returns scores, not the fitted
    # centers the C side needs -- refit directly with identical KMeans
    # params (n_clusters=16, n_init=10, random_state=seed) so this can't
    # silently drift from what auc_pauc.py's score_knn_clustered does.
    kmeans = KMeans(n_clusters=16, n_init=10, random_state=seed).fit(train_emb)
    clusters = kmeans.cluster_centers_
    knn16_scores = np.sort(cdist(test_emb, clusters), axis=1)[:, 0]

    auc_euclidean = roc_auc_score(test_labels, euclidean_scores)
    pauc_euclidean = roc_auc_score(test_labels, euclidean_scores, max_fpr=0.1)
    auc_knn16 = roc_auc_score(test_labels, knn16_scores)
    pauc_knn16 = roc_auc_score(test_labels, knn16_scores, max_fpr=0.1)

    # Threshold: parametric fitting by default (findings/140 sec 9's actual
    # recommendation), EVT available via --threshold-method evt for
    # comparison. Fit on THIS scheme's calib_normal scores either way.
    calib_euclidean_scores, _ = DISTANCE_HEADS["euclidean"](train_emb, calib_emb, seed)
    calib_knn16_scores = np.sort(cdist(calib_emb, clusters), axis=1)[:, 0]

    threshold_fn = {"evt": evt_threshold, "parametric": parametric_threshold}[args.threshold_method]
    threshold_euclidean, threshold_info_euclidean = threshold_fn(calib_euclidean_scores, args.target_far)
    threshold_knn16, threshold_info_knn16 = threshold_fn(calib_knn16_scores, args.target_far)

    # Diagnostic worth checking regardless of method chosen: findings/140
    # sec 7 found EVT shape >= 0.676 predicts collapse with zero exceptions
    # (48/48 configurations tested there). Surfaced here so a bad EVT fit
    # is visible immediately rather than only showing up as a puzzlingly
    # low recall three steps later.
    if args.threshold_method == "evt":
        for label, info in [("euclidean", threshold_info_euclidean), ("knn_16", threshold_info_knn16)]:
            shape = info.get("shape")
            if shape is not None and shape >= 0.676:
                print(f"  WARNING: {label}'s EVT shape={shape:.4f} is at or above the "
                      f"0.676 collapse threshold findings/140 sec 7 established (48/48 "
                      f"historical exceptions) -- expect degenerate recall. Consider "
                      f"--threshold-method parametric.")

    metrics_euclidean = secondary_metrics(euclidean_scores, test_labels, threshold_euclidean)
    metrics_knn16 = secondary_metrics(knn16_scores, test_labels, threshold_knn16)

    print(f"\n{scheme_desc}  (n_test={len(test_labels)}, anomalies={int(test_labels.sum())}, "
          f"target_far={args.target_far})")
    print(f"  euclidean  AUC={auc_euclidean:.4f} pAUC={pauc_euclidean:.4f} "
          f"threshold={threshold_euclidean:.6f} "
          f"P={metrics_euclidean['precision']:.3f} R={metrics_euclidean['recall']:.3f} "
          f"A={metrics_euclidean['accuracy']:.3f} F1={metrics_euclidean['f1']:.3f}")
    print(f"  knn_16     AUC={auc_knn16:.4f} pAUC={pauc_knn16:.4f} "
          f"threshold={threshold_knn16:.6f} "
          f"P={metrics_knn16['precision']:.3f} R={metrics_knn16['recall']:.3f} "
          f"A={metrics_knn16['accuracy']:.3f} F1={metrics_knn16['f1']:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = {
        "scheme": scheme_desc,
        "target_far": args.target_far,
        "threshold_method": args.threshold_method,
        "n_train": len(train_emb), "n_test": len(test_emb), "n_calib_normal": len(calib_emb),
        "euclidean": {"auc": float(auc_euclidean), "pauc": float(pauc_euclidean),
                     "threshold": threshold_euclidean, "threshold_info": threshold_info_euclidean,
                     **metrics_euclidean},
        "knn_16": {"auc": float(auc_knn16), "pauc": float(pauc_knn16),
                  "threshold": threshold_knn16, "threshold_info": threshold_info_knn16,
                  **metrics_knn16},
    }

    if args.scheme == "combined":
        # Float reference, float threshold -- this scheme's pooled embedding
        # is a real-valued average of already-quantized values, NOT a
        # guaranteed exact int8 round trip (its GetPooled has no final
        # re-quantize step), so it can't use the pure-int32 head.
        header = """#pragma once

#include "ssm_weights.h"

#define SSM_KNN16_N_CLUSTERS 16

extern const float ssm_ref_centroid[SSM_D_MODEL];
extern const float ssm_ref_clusters[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL];
extern const float ssm_threshold_euclidean;
extern const float ssm_threshold_knn16;
"""
        (out_dir / "ssm_head_ref.h").write_text(header)

        body_lines = [
            '#include "ssm_head_ref.h"',
            "",
            c_array_1d("ssm_ref_centroid", centroid),
            "",
            c_array_2d("ssm_ref_clusters", clusters),
            "",
            f"const float ssm_threshold_euclidean = {format_c_float(threshold_euclidean)};",
            f"const float ssm_threshold_knn16 = {format_c_float(threshold_knn16)};",
        ]
        (out_dir / "ssm_head_ref.c").write_text("\n".join(body_lines) + "\n")

        for fname in HEAD_FILES:
            shutil.copy(HEAD_SRC_DIR_FLOAT / fname, out_dir / fname)

        with open(out_dir / "head_export_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=4)

        print(f"\nWrote {out_dir/'ssm_head_ref.h'}, {out_dir/'ssm_head_ref.c'}, "
              f"{', '.join(str(out_dir/f) for f in HEAD_FILES)}, "
              f"and {out_dir/'head_export_diagnostics.json'}")

    else:  # true_int8
        # Quantize the centroid/clusters to the SAME scale the embedding
        # itself lives on -- ranges['final_norm_output'], the exact value
        # export_true_int8_weights.py already baked in as ssm_final_norm_scale
        # for this h-width. Never re-derived independently: if this ever
        # disagreed with the weight export's own scale, the reference
        # vectors and the embedding would silently live in two different
        # domains and every comparison would be meaningless.
        final_norm_scale = ranges["final_norm_output"]

        centroid_q = quantize_ref_int8(centroid, final_norm_scale, "centroid")
        clusters_q = quantize_ref_int8(clusters, final_norm_scale, "cluster centers")

        # Squared thresholds, precomputed once here so the device never
        # needs a sqrt to make a decision (sqrt is monotonic; comparing
        # sum-of-squares directly against a squared threshold gives an
        # identical decision, since every candidate shares one scale).
        threshold_euclidean_sumsq = int(round((threshold_euclidean / final_norm_scale) ** 2))
        threshold_knn16_sumsq = int(round((threshold_knn16 / final_norm_scale) ** 2))

        header = """#pragma once

#include <stdint.h>
#include "ssm_weights.h"

#define SSM_KNN16_N_CLUSTERS 16

/* Quantized to ssm_final_norm_scale's domain -- the SAME scale the pooled
 * embedding itself lives on. Not a general-purpose float reference; see
 * ssm_distance_head_true_int8.h. */
extern const int8_t ssm_ref_centroid_q[SSM_D_MODEL];
extern const int8_t ssm_ref_clusters_q[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL];

/* Precomputed as round((threshold_real / ssm_final_norm_scale)^2) at
 * export time -- compared directly against euclidean_sumsq/knn16_sumsq,
 * no sqrt needed at decision time. */
extern const int32_t ssm_threshold_euclidean_sumsq;
extern const int32_t ssm_threshold_knn16_sumsq;
"""
        (out_dir / "ssm_head_ref.h").write_text(header)

        body_lines = [
            '#include "ssm_head_ref.h"',
            "",
            c_int8_array_1d("ssm_ref_centroid_q", centroid_q),
            "",
            c_int8_array_2d("ssm_ref_clusters_q", clusters_q),
            "",
            f"const int32_t ssm_threshold_euclidean_sumsq = {threshold_euclidean_sumsq};",
            f"const int32_t ssm_threshold_knn16_sumsq = {threshold_knn16_sumsq};",
        ]
        (out_dir / "ssm_head_ref.c").write_text("\n".join(body_lines) + "\n")

        for fname in HEAD_FILES:
            shutil.copy(HEAD_SRC_DIR_TRUE_INT8 / fname, out_dir / fname)

        diagnostics["final_norm_scale"] = final_norm_scale
        diagnostics["euclidean"]["threshold_sumsq"] = threshold_euclidean_sumsq
        diagnostics["knn_16"]["threshold_sumsq"] = threshold_knn16_sumsq
        with open(out_dir / "head_export_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=4)

        print(f"\nWrote {out_dir/'ssm_head_ref.h'}, {out_dir/'ssm_head_ref.c'}, "
              f"{', '.join(str(out_dir/f) for f in HEAD_FILES)}, "
              f"and {out_dir/'head_export_diagnostics.json'}")
        print(f"threshold_euclidean_sumsq={threshold_euclidean_sumsq}  "
              f"threshold_knn16_sumsq={threshold_knn16_sumsq}")


if __name__ == "__main__":
    main()