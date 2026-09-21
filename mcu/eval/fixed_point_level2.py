r"""
Level 2 of the two-level rule: do the fixed-point log-mel frames change the
model's embeddings, scores, and AUC?

Stage 2 of 2. Stage 1 (test_harness/fixed_point_export_logmel.py, Windows)
writes the log-mel frames. This script runs in the project environment
(torch, src.*) from the PROJECT ROOT:

    python -m mcu.eval.fixed_point_level2
    python -m mcu.eval.fixed_point_level2 --backbones fp32 int8_h8 --max-clips 40
    python -m mcu.eval.fixed_point_level2 --backbones fp32          # fast first pass

What it does
  * For each backbone (fp32, int8_h8, int8_h16) it builds the embeddings of
    the test clips from each feature set: 'cache' (the project's cached
    log-mel), 'ref64', 'ref32', and the fixed-point variants.
  * It scores them with the DEPLOYED head. The reference vector and the
    threshold are parsed from the setup folder's ssm_head_ref.c (the fp32
    head of the matching backbone). Nothing is refitted.
  * Self-check: with the cached features it must reproduce the setup's
    offline/scores.csv.
  * It reports AUC, pAUC, the score and embedding change against ref64, and
    the number of decisions that flip at the baked threshold.

Verdict per feature set (drop = AUC of ref64 minus AUC of the set):
    PASS      drop <= --auc-tol (default 0.001)
    MARGINAL  drop <= SEED_SD = 0.0132 (findings 210 and 250: not
              distinguishable from seed-to-seed training noise)
    FAIL      larger
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from src.config import load_config_by_name, PROJECT_ROOT
from src.features.stats import compute_normalization_stats
from src.features.baselines import apply_normalization
from src.models.backbone import SSMBackbone
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import (
    calibrate_missing_ranges, prepare_quantized_model, quantized_forward,
)
from mcu.export_deploy_matrix import load_resolved_fold
from mcu.compute_offline_metrics import parse_c_float_array, parse_c_2d_array

DEFAULT_FEATURES_ROOT = ("/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/"
                         "test_harness/output_fixed_point_level2")
SEED_SD = 0.0132  # findings 210 and 250

BACKBONES = {
    "fp32": {"setup": "full_fp32_{head}_fp32", "kind": "fp32"},
    "int8_h8": {"setup": "true_int8_h8_{head}_fp32", "kind": "int8", "h_width": "int8"},
    "int8_h16": {"setup": "true_int8_h16_{head}_fp32", "kind": "int8", "h_width": "int16"},
}
DEFAULT_SETS = ["cache", "ref64", "ref32", "q31_plain", "q31_gain", "q15_plain", "q15_gain"]


def balanced_indices(labels, n):
    normal = np.where(labels == 0)[0]
    anomaly = np.where(labels == 1)[0]
    n_anom = min(len(anomaly), n // 2)
    n_norm = min(len(normal), n - n_anom)
    return np.array(sorted(list(normal[:n_norm]) + list(anomaly[:n_anom])))


def parse_threshold(text, name):
    import re
    m = re.search(rf"\b{name}\s*=\s*([^;]+);", text)
    if not m:
        raise ValueError(f"{name} not found")
    return float(m.group(1).strip().rstrip("fF"))


def load_head_reference(setup_dir, head):
    text = (setup_dir / "ssm_head_ref.c").read_text()
    if head == "euclidean":
        centroid = parse_c_float_array(text, "ssm_ref_centroid")
        thr = parse_threshold(text, "ssm_threshold_euclidean")
        c = centroid.astype(np.float64)
        return (lambda e: np.linalg.norm(e - c, axis=1)), thr
    clusters = parse_c_2d_array(text, "ssm_ref_clusters", dtype="float").astype(np.float64)
    thr = parse_threshold(text, "ssm_threshold_knn16")
    return (lambda e: np.min(cdist(e, clusters), axis=1)), thr


def make_fp32_embedder(cfg, base_dir, mean, std):
    model = SSMBackbone(**cfg["model"])
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.pooling = "mean"

    def embed(feats):
        shapes = {a.shape for a in feats}
        if len(shapes) != 1:
            raise ValueError(f"fp32 batching needs one shape, got {shapes}")
        X = np.stack([apply_normalization(a, mean, std) for a in feats]).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                batch = torch.from_numpy(X[i:i + 128]).float()
                out.append(model(batch, mode="pooled", quantizer=None).cpu().numpy())
        return np.concatenate(out, axis=0)
    return embed


def make_int8_embedder(cfg, base_dir, fold, mean, std, h_width, hook_cache):
    m = cfg["model"]
    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dt_rank = max(1, d_model // 16)

    ranges = load_ranges_scales(base_dir)
    if "scales" not in hook_cache:
        model_for_hooks = SSMBackbone(**m)
        ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
        model_for_hooks.load_state_dict(ckpt["model"])
        model_for_hooks.eval()
        print("  calibrating true-int8 missing ranges (once)...")
        hook_cache["scales"] = calibrate_missing_ranges(
            model_for_hooks, fold, mean, std, n_layers, dt_rank)
    hook_scales = hook_cache["scales"]

    model = SSMBackbone(**m)
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()
    weights, scales, luts = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, h_width)

    def embed(feats):
        out = np.zeros((len(feats), d_model))
        started = time.time()
        for i, a in enumerate(feats):
            x = apply_normalization(a.astype(np.float32), mean, std).astype(np.float32)
            out[i] = quantized_forward(x, sd, weights, scales, luts, ranges,
                                       d_model, d_inner, d_state, d_conv, n_layers, h_width)
            if (i + 1) % 20 == 0 or i + 1 == len(feats):
                elapsed = time.time() - started
                print(f"      {i + 1}/{len(feats)} clips, {elapsed / 60:.1f} min, "
                      f"ETA {elapsed / (i + 1) * (len(feats) - i - 1) / 60:.1f} min")
        return out
    return embed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="f4cd557b7e3b.yaml")
    parser.add_argument("--held-out-case", type=int, default=1)
    parser.add_argument("--model-hash", default="16662b29beb3")
    parser.add_argument("--features-root", default=DEFAULT_FEATURES_ROOT,
                        help="output folder of fixed_point_export_logmel.py")
    parser.add_argument("--backbones", nargs="+", default=["fp32", "int8_h8"],
                        choices=list(BACKBONES))
    parser.add_argument("--head", choices=["euclidean", "knn16"], default="euclidean")
    parser.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    parser.add_argument("--reference", default="ref64",
                        help="feature set that the others are compared with")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="class-balanced subset of the exported clips")
    parser.add_argument("--auc-tol", type=float, default=0.001)
    parser.add_argument("--reuse", action="store_true",
                        help="reuse embeddings saved by an earlier run")
    parser.add_argument("--deploy-root", default="mcu/deploy")
    args = parser.parse_args()

    features_root = Path(args.features_root)
    out_dir = features_root / "level2_results"
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    if not m["selective"] or m["discretization"] != "euler":
        raise ValueError("this script handles selective, euler configs only")
    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    fold = load_resolved_fold(base_dir)
    mean, std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])

    test = fold["test"].reset_index(drop=True)
    stems = [Path(p).stem for p in test["path"]]
    labels_all = (test["label"].values == "anomaly").astype(int)

    # Keep the clips that have every requested feature set exported.
    sets = [s for s in args.sets]
    if args.reference not in sets:
        sets.append(args.reference)
    keep = []
    for i, stem in enumerate(stems):
        ok = all(s == "cache" or (features_root / "logmel" / s / f"{stem}.npy").is_file()
                 for s in sets)
        if ok:
            keep.append(i)
    keep = np.array(keep)
    if len(keep) == 0:
        raise SystemExit("No clip has all requested feature sets. Run stage 1 first.")
    if args.max_clips and len(keep) > args.max_clips:
        keep = keep[balanced_indices(labels_all[keep], args.max_clips)]
    labels = labels_all[keep]
    clip_stems = [stems[i] for i in keep]
    cache_paths = [test["cache_path"].iloc[i] for i in keep]
    print(f"{len(keep)} clips ({int(labels.sum())} anomaly, {int((1 - labels).sum())} normal)")
    if labels.min() == labels.max():
        raise SystemExit("The clip subset has one class only. AUC needs both.")

    def load_set(name):
        if name == "cache":
            return [np.load(p).astype(np.float32) for p in cache_paths]
        return [np.load(features_root / "logmel" / name / f"{s}.npy").astype(np.float32)
                for s in clip_stems]

    features = {name: load_set(name) for name in sets}
    for name in sets:
        shapes = {a.shape for a in features[name]}
        print(f"  set {name}: frame shapes {sorted(shapes)[:3]}")

    all_rows = []
    hook_cache = {}
    for backbone in args.backbones:
        spec = BACKBONES[backbone]
        setup_id = spec["setup"].format(head=args.head)
        setup_dir = (PROJECT_ROOT / args.deploy_root / f"case{args.held_out_case}" /
                     args.model_hash / "setups" / setup_id)
        if not (setup_dir / "ssm_head_ref.c").is_file():
            print(f"\n{backbone}: no ssm_head_ref.c in {setup_dir}, skipped")
            continue
        print(f"\n=== backbone {backbone} (head reference from {setup_id}) ===")
        scorer, threshold = load_head_reference(setup_dir, args.head)

        if spec["kind"] == "fp32":
            embedder = make_fp32_embedder(cfg, base_dir, mean, std)
        else:
            embedder = make_int8_embedder(cfg, base_dir, fold, mean, std,
                                          spec["h_width"], hook_cache)

        embeddings = {}
        for name in sets:
            emb_file = out_dir / "embeddings" / f"{backbone}__{name}__{len(keep)}.npy"
            if args.reuse and emb_file.is_file():
                embeddings[name] = np.load(emb_file)
                print(f"  {name}: reused {emb_file.name}")
                continue
            print(f"  embedding feature set '{name}'...")
            embeddings[name] = embedder(features[name])
            np.save(emb_file, embeddings[name])

        scores = {name: scorer(embeddings[name]) for name in sets}

        # Self-check against the offline scores of the deployed setup.
        offline_csv = setup_dir / "offline" / "scores.csv"
        if "cache" in scores and offline_csv.is_file():
            offline = pd.read_csv(offline_csv).set_index("clip_name")["score"]
            common = [s for s in clip_stems if s in offline.index]
            if common:
                mine = np.array([scores["cache"][clip_stems.index(s)] for s in common])
                theirs = offline.loc[common].values
                rel = np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-12))
                print(f"  self-check: cached-feature scores against {offline_csv.name}: "
                      f"max relative difference {rel:.2e} "
                      f"({'OK' if rel < 1e-3 else 'MISMATCH - check the setup'})")
        else:
            print("  self-check skipped (no cache set or no offline/scores.csv)")

        ref_scores = scores[args.reference]
        ref_emb = embeddings[args.reference]
        ref_auc = roc_auc_score(labels, ref_scores)
        ref_decision = ref_scores > threshold
        print(f"  baked threshold {threshold:.6g}; reference '{args.reference}' "
              f"AUC {ref_auc:.4f}")

        for name in sets:
            s = scores[name]
            auc = roc_auc_score(labels, s)
            pauc = roc_auc_score(labels, s, max_fpr=0.1)
            rel_score = np.abs(s - ref_scores) / np.maximum(np.abs(ref_scores), 1e-12)
            rel_emb = (np.linalg.norm(embeddings[name] - ref_emb, axis=1) /
                       np.maximum(np.linalg.norm(ref_emb, axis=1), 1e-12))
            flips = int(np.sum((s > threshold) != ref_decision))
            drop = ref_auc - auc
            if name == args.reference:
                verdict = "reference"
            elif name in ("cache", "ref32"):
                verdict = "context"
            elif drop <= args.auc_tol:
                verdict = "PASS"
            elif drop <= SEED_SD:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"
            all_rows.append({
                "backbone": backbone, "set": name, "auc": auc, "pauc": pauc,
                "auc_drop": drop, "max_rel_score": float(rel_score.max()),
                "mean_rel_score": float(rel_score.mean()),
                "max_rel_emb": float(rel_emb.max()), "flips": flips,
                "n_clips": len(keep), "verdict": verdict,
            })

        pd.DataFrame({"clip": clip_stems, "label": labels,
                      **{name: scores[name] for name in sets}}
                     ).to_csv(out_dir / f"scores_{backbone}.csv", index=False)

    if not all_rows:
        raise SystemExit("Nothing was evaluated.")
    table = pd.DataFrame(all_rows)
    table.to_csv(out_dir / "level2_summary.csv", index=False)

    print("\n=== Level 2 summary ===")
    print(f"AUC drop is (reference AUC - set AUC). PASS <= {args.auc_tol}, "
          f"MARGINAL <= {SEED_SD} (seed noise), FAIL above.\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nResults are in {out_dir}")


if __name__ == "__main__":
    main()