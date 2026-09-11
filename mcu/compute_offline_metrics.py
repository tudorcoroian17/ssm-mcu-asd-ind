r"""
Offline (PC-side) AUC/pAUC/precision/recall/accuracy/F1 computation for
every deployment setup export_deploy_matrix.py produces -- the "test on
PC before burning hardware hours" companion. Writes, per setup, under
mcu/deploy/case<N>/<model_hash>/setups/<setup_id>/offline/:

    metrics.json           AUC, pAUC, and all 14 threshold methods' P/R/A/F1
    embeddings/<clip>.npy  same per-clip format as run_deployment_test.py's
                            online capture
    scores/<clip>.csv      same columns as run_deployment_test.py's online
                            capture, scored under the ACTIVE (baked-in)
                            threshold method only
    scores.csv             combined, whole-run version of the above

REUSE, NOT REIMPLEMENTATION: imports build_setup_matrix, load_resolved_fold,
build_fake_quant_embeddings, build_true_int8_embeddings, fit_head,
compute_all_thresholds, quantize_ref_int8, DEFAULT_THRESHOLD_METHOD, and
KNN16_N_CLUSTERS directly from export_deploy_matrix.py (run from the same
mcu/ directory so this resolves as a plain sibling-module import -- no
package structure needed). This guarantees the offline numbers use
IDENTICAL logic to what actually got baked into each setup's C files, and
the same per-backbone caching: embeddings and head fits are each computed
once per distinct backbone config, not once per setup (7 distinct configs
across all 18 setups, exactly as export_deploy_matrix.py itself caches).

SPLIT SOURCE: load_resolved_fold(base_dir) -- the SAME
runs/case<N>/<model_hash>/embeddings/manifest.csv 'used_in' column
run_deployment_test.py's load_split() reads, and which export_deploy_matrix.py
was fixed to read too. This is what makes an offline/online comparison
meaningful: both scripts test the identical clip set. If you're running
this against setup folders exported BEFORE that fix, re-run
export_deploy_matrix.py first -- the parity check below will otherwise
flag every setup as mismatched, which is the correct behavior, not a bug
in this script.

INT8-HEAD SCORING (4 of the 18 setups: true_int8 x {euclidean,knn16} x
head_precision=int8): offline/scores' score/decision replicate the EXACT
device-side int32 arithmetic path (quantize embedding to int8 at
final_norm_scale via quantize_ref_int8 -- the SAME function used to
quantize the reference vectors at export time -- then int64
sum-of-squared-differences against the quantized reference(s), sqrt+scale
for the reported score, integer sumsq compared against the integer
threshold for the decision). This is deliberately NOT the plain float
score used for this setup's own AUC/pAUC: AUC measures the algorithm's
ranking quality independent of deployment precision, but "does this score
match the online capture" needs the device's actual arithmetic. For the
14 fp32-head setups these two are identical, so this distinction is
invisible there.

PARITY CHECK: for every already-exported setup, re-parses its OWN
ssm_head_ref.c (regex, since the format is fully controlled by
export_deploy_matrix.py's own c_array_1d_public/c_int8_array_1d_public
emitters) for the ACTIVE head's reference array (centroid for a
euclidean-active setup, clusters for a knn16-active setup -- the inactive
one is always a zero placeholder in that file, never meaningful) and
diffs it against this run's own freshly-computed version. A mismatch
means the deployed C files are stale relative to this offline
recomputation.

Usage:
    python mcu/compute_offline_metrics.py \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3
    # optional: --only w_all_pertensor_euclidean_fp32,true_int8_h8_knn16_int8
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config_by_name, PROJECT_ROOT
from src.models.backbone import SSMBackbone
from src.features.stats import compute_normalization_stats
from src.eval.thresholds import secondary_metrics

from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import calibrate_missing_ranges

from mcu.export_deploy_matrix import (
    build_setup_matrix, load_resolved_fold, build_fake_quant_embeddings,
    build_true_int8_embeddings, fit_head, compute_all_thresholds,
    quantize_ref_int8, DEFAULT_THRESHOLD_METHOD, KNN16_N_CLUSTERS,
)


def parse_c_float_array(text, var_name):
    """Extracts a `const float <var_name>[...] = { ... };` literal --
    matches c_array_1d_public's exact single-line output format."""
    m = re.search(rf'\b{re.escape(var_name)}\s*\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}', text)
    if not m:
        return None
    values = [v.strip() for v in m.group(1).split(",") if v.strip()]
    return np.array([float(v.rstrip("fF")) for v in values], dtype=np.float32)


def parse_c_int_array(text, var_name):
    """Extracts a `const int8_t <var_name>[...] = { ... };` literal --
    matches c_int8_array_1d_public's exact single-line output format."""
    m = re.search(rf'\b{re.escape(var_name)}\s*\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}', text)
    if not m:
        return None
    values = [v.strip() for v in m.group(1).split(",") if v.strip()]
    return np.array([int(v) for v in values], dtype=np.int64)


def parse_c_2d_array(text, var_name, dtype):
    """Extracts a `const <T> <var_name>[R][C] = { {..}, {..}, ... };`
    literal -- matches c_array_2d/c_int8_array_2d's multi-line output
    format (DOTALL, since rows are newline-separated)."""
    m = re.search(rf'\b{re.escape(var_name)}\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=\s*\{{(.*?)\n\}};',
                  text, re.DOTALL)
    if not m:
        return None
    row_texts = re.findall(r'\{([^{}]*)\}', m.group(1))
    rows = []
    for row in row_texts:
        values = [v.strip() for v in row.split(",") if v.strip()]
        if dtype == "float":
            rows.append([float(v.rstrip("fF")) for v in values])
        else:
            rows.append([int(v) for v in values])
    return np.array(rows, dtype=np.float32 if dtype == "float" else np.int64)


def check_ref_parity(setup, ref, final_norm_scale):
    """Diffs THIS run's own freshly-computed reference (and, for int8
    heads, freshly re-quantized reference) against what's already sitting
    in the setup's deployed ssm_head_ref.c. Returns (ok: bool, detail: str)."""
    head_ref_c = Path(setup["_setup_dir"]) / "ssm_head_ref.c"
    if not head_ref_c.is_file():
        return False, f"ssm_head_ref.c not found at {head_ref_c}"
    text = head_ref_c.read_text()

    if setup["head_precision"] == "int8":
        if setup["head"] == "euclidean":
            deployed = parse_c_int_array(text, "ssm_ref_centroid_q")
            fresh = quantize_ref_int8(ref["centroid"], final_norm_scale)
        else:
            deployed = parse_c_2d_array(text, "ssm_ref_clusters_q", dtype="int")
            fresh = quantize_ref_int8(ref["clusters"], final_norm_scale)
        if deployed is None:
            return False, "could not parse int8 reference array from ssm_head_ref.c"
        if deployed.shape != fresh.shape:
            return False, f"shape mismatch: deployed {deployed.shape} vs fresh {fresh.shape}"
        if not np.array_equal(deployed, fresh.astype(np.int64)):
            n_diff = int(np.sum(deployed != fresh.astype(np.int64)))
            return False, f"{n_diff} of {deployed.size} int8 reference values differ"
        return True, "exact match"
    else:
        if setup["head"] == "euclidean":
            deployed = parse_c_float_array(text, "ssm_ref_centroid")
            fresh = np.asarray(ref["centroid"], dtype=np.float32)
        else:
            deployed = parse_c_2d_array(text, "ssm_ref_clusters", dtype="float")
            fresh = np.asarray(ref["clusters"], dtype=np.float32)
        if deployed is None:
            return False, "could not parse float reference array from ssm_head_ref.c"
        if deployed.shape != fresh.shape:
            return False, f"shape mismatch: deployed {deployed.shape} vs fresh {fresh.shape}"
        max_diff = float(np.max(np.abs(deployed - fresh)))
        if max_diff > 1e-4:
            return False, f"max abs diff {max_diff:.6e} exceeds float-formatting tolerance"
        return True, f"match within formatting tolerance (max abs diff {max_diff:.2e})"


def score_int8_device_path(embedding_real, ref, final_norm_scale, head, threshold_sumsq):
    """Replicates SSMDistanceHead_ScoreTrueInt8's exact int32 arithmetic
    path in Python: quantize_ref_int8 (the SAME function export_deploy_matrix.py
    uses to quantize reference vectors at export time) applied to the
    embedding itself, then int64 sum-of-squared-differences (wide enough
    to never overflow, mirroring the C side's int32 being provably safe
    for these dimensions), sqrt+scale for the reported score, integer
    sumsq compared against the integer threshold for the decision.
    threshold_sumsq is the SAME int(round((threshold/scale)**2)) formula
    export_deploy_matrix.py's own emit_head_ref_and_module uses."""
    embedding_q = quantize_ref_int8(embedding_real, final_norm_scale).astype(np.int64)

    if head == "euclidean":
        centroid_q = quantize_ref_int8(ref["centroid"], final_norm_scale).astype(np.int64)
        sumsq = int(np.sum((embedding_q - centroid_q) ** 2))
    else:
        clusters_q = quantize_ref_int8(ref["clusters"], final_norm_scale).astype(np.int64)
        sumsq = int(np.min(np.sum((embedding_q[None, :] - clusters_q) ** 2, axis=1)))

    score = float(np.sqrt(sumsq) * final_norm_scale)
    decision = int(sumsq > threshold_sumsq)
    return score, decision, sumsq


def write_csv_row(csv_path, row):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--deploy-root", default="mcu/deploy")
    parser.add_argument("--only", default=None,
                        help="comma-separated setup identifiers (default: all 18)")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    seed = cfg["seed"]
    if not m["selective"]:
        raise ValueError("this matrix only handles selective=True")
    if m["discretization"] != "euler":
        raise ValueError(f"true-int8 backbone assumes euler; config says {m['discretization']!r}")

    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dt_rank = max(1, d_model // 16)
    dims = (d_model, d_state, d_inner, d_conv, n_layers, dt_rank)

    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    if not (base_dir / "ckpt.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {base_dir/'ckpt.pt'}")

    # Same source as export_deploy_matrix.py post-fix and run_deployment_test.py --
    # this identity is what makes an offline/online comparison meaningful.
    fold = load_resolved_fold(base_dir)
    test_rows = fold["test"].reset_index(drop=True)
    test_labels = (test_rows["label"].values == "anomaly").astype(int)
    clip_names = [Path(p).stem for p in test_rows["path"]]
    print(f"Test split: {len(test_rows)} clips "
         f"({int((test_labels == 0).sum())} normal, {int((test_labels == 1).sum())} anomaly)")

    norm_mean, norm_std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    norm_stats = (norm_mean, norm_std)

    setups = build_setup_matrix()
    if args.only:
        wanted = set(args.only.split(","))
        setups = [s for s in setups if s["identifier"] in wanted]
        if not setups:
            raise ValueError(f"--only matched no setups; valid ids: "
                             f"{[s['identifier'] for s in build_setup_matrix()]}")

    # Shared true-int8 calibration, same pattern as export_deploy_matrix.py:
    # ranges and hook scales are identical across both h-widths.
    ti_ranges = None
    ti_hook_scales = None
    if any(s["family"] == "true_int8" for s in setups):
        ti_ranges = load_ranges_scales(base_dir)
        model_for_hooks = SSMBackbone(**m)
        ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
        model_for_hooks.load_state_dict(ckpt["model"])
        model_for_hooks.eval()
        print("Calibrating true-int8 missing ranges (once, shared across h-widths)...")
        ti_hook_scales = calibrate_missing_ranges(model_for_hooks, fold, norm_mean, norm_std,
                                                  n_layers, dt_rank)

    emb_cache = {}

    def get_embeddings(setup):
        key = setup["backbone_key"]
        if key in emb_cache:
            return emb_cache[key]
        print(f"  computing embeddings for backbone '{key}'...")
        if setup["family"] == "fake_quant":
            emb = build_fake_quant_embeddings(
                cfg, base_dir, fold, norm_stats,
                setup["weight_mode"], setup["granularity"], setup["activation_group"])
        else:
            emb = build_true_int8_embeddings(
                cfg, base_dir, fold, norm_stats, dims, ti_ranges, ti_hook_scales, setup["h_width"])
        emb_cache[key] = emb
        return emb

    head_cache = {}

    def get_head(setup, train_emb):
        key = (setup["backbone_key"], setup["head"])
        if key not in head_cache:
            head_cache[key] = fit_head(setup["head"], train_emb, seed)
        return head_cache[key]

    deploy_root = PROJECT_ROOT / args.deploy_root
    setups_root = deploy_root / f"case{args.held_out_case}" / args.model_hash / "setups"
    all_identifiers = [s["identifier"] for s in build_setup_matrix()]

    for setup in setups:
        combo_number = all_identifiers.index(setup["identifier"]) + 1
        ident = setup["identifier"]
        setup_dir = setups_root / ident
        if not setup_dir.is_dir():
            print(f"\n[{combo_number:2d}/18] {ident}: SKIP -- not yet exported "
                 f"by export_deploy_matrix.py")
            continue
        setup["_setup_dir"] = setup_dir  # threaded through to check_ref_parity
        print(f"\n[{combo_number:2d}/18] {ident}")

        offline_dir = setup_dir / "offline"
        if offline_dir.exists():
            # Idempotent per run: without this, re-running this script for
            # a setup already processed would silently APPEND a duplicate
            # row to every per-clip CSV (write_csv_row only skips the
            # header when the file exists, it doesn't overwrite), quietly
            # corrupting exactly the files this exists to keep trustworthy.
            shutil.rmtree(offline_dir)
        (offline_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (offline_dir / "scores").mkdir(parents=True, exist_ok=True)
        combined_scores_csv = offline_dir / "scores.csv"

        emb = get_embeddings(setup)
        train_emb, test_emb = emb["train"], emb["test"]

        ref, scorer = get_head(setup, train_emb)
        test_scores = scorer(test_emb)
        calib_scores = scorer(emb["calib_normal"])

        auc = float(roc_auc_score(test_labels, test_scores))
        pauc = float(roc_auc_score(test_labels, test_scores, max_fpr=0.1))

        thresholds = compute_all_thresholds(calib_scores, seed)
        metrics_by_threshold = {}
        for method_name, (thr, info) in thresholds.items():
            mm = secondary_metrics(test_scores, test_labels, float(thr))
            metrics_by_threshold[method_name] = {
                "threshold": float(thr), "info": info,
                "metrics": {k: mm[k] for k in ("precision", "recall", "accuracy", "f1")},
            }
        default_thr = metrics_by_threshold[DEFAULT_THRESHOLD_METHOD]["threshold"]

        final_norm_scale = None
        default_thr_sumsq = None
        if setup["head_precision"] == "int8":
            final_norm_scale = ti_ranges["final_norm_output"]
            default_thr_sumsq = int(round((default_thr / final_norm_scale) ** 2))

        parity_ok, parity_detail = check_ref_parity(setup, ref, final_norm_scale)
        print(f"    parity vs deployed ssm_head_ref.c: "
             f"{'OK' if parity_ok else 'MISMATCH'} ({parity_detail})")

        # --- per-clip embeddings + scores, same shape/columns as the online capture ---
        for i, clip_name in enumerate(clip_names):
            np.save(offline_dir / "embeddings" / f"{clip_name}.npy",
                   np.asarray(test_emb[i], dtype=np.float32))

            if setup["head_precision"] == "int8":
                score, decision, sumsq = score_int8_device_path(
                    test_emb[i], ref, final_norm_scale, setup["head"], default_thr_sumsq)
            else:
                score = float(test_scores[i])
                decision = int(score > default_thr)
                sumsq = None

            row = {
                "clip_name": clip_name, "true_label": test_rows.iloc[i]["label"],
                "head": setup["head"], "head_precision": setup["head_precision"],
                "score": score, "decision": decision, "sumsq": sumsq,
            }
            write_csv_row(offline_dir / "scores" / f"{clip_name}.csv", row)
            write_csv_row(combined_scores_csv, row)

        # --- metrics.json ---
        metrics = {
            "identifier": ident,
            "config": args.config, "held_out_case": args.held_out_case,
            "model_hash": args.model_hash,
            "n_test_clips": len(test_rows),
            "default_threshold_method": DEFAULT_THRESHOLD_METHOD,
            "parity_check_against_deployed_c": {"ok": parity_ok, "detail": parity_detail},
            "auc": auc, "pauc": pauc,
            "thresholds": metrics_by_threshold,
        }
        with open(offline_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4, default=str)

        default_m = metrics_by_threshold[DEFAULT_THRESHOLD_METHOD]["metrics"]
        print(f"    AUC={auc:.4f} pAUC={pauc:.4f}  default '{DEFAULT_THRESHOLD_METHOD}' "
             f"P={default_m['precision']:.3f} R={default_m['recall']:.3f} "
             f"A={default_m['accuracy']:.3f} F1={default_m['f1']:.3f}")

    print(f"\nDone. Offline artifacts written under each setup's offline/ subfolder.")


if __name__ == "__main__":
    main()