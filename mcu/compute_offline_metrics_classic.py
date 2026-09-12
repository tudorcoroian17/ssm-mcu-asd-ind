r"""
Offline (PC-side) metrics for every setup export_deploy_matrix_classic.py
produces -- the classic (selective=False) companion to
mcu/compute_offline_metrics.py. Writes the same artifacts, in the same
per-setup layout:

    mcu/deploy/case<N>/<model_hash>/setups/<setup_id>/offline/
        metrics.json           AUC, pAUC, and all 14 threshold methods' P/R/A/F1
        embeddings/<clip>.npy  same per-clip format as the online capture
        scores/<clip>.csv      same columns as the online capture
        scores.csv             combined, whole-run version

WHY A SEPARATE FILE: mcu/compute_offline_metrics.py is bound to the
selective branch in five places -- its selective=True guard, dt_rank and
the six-element dims tuple, calibrate_missing_ranges (which hooks
block.x_proj and block.dt_proj, neither of which exists on a classic
block), build_setup_matrix plus the two selective embedding builders, and
the hardcoded /18 in its progress prints. It is NOT modified by this file
and does not import from it; the dependency runs one way only, exactly as
export_deploy_matrix_classic.py relates to export_deploy_matrix.py.

WHAT IS REUSED RATHER THAN REIMPLEMENTED: check_ref_parity,
score_int8_device_path and write_csv_row come straight from
compute_offline_metrics.py, unchanged. The parity check in particular
needs no classic variant: ssm_head_ref.c is emitted by
emit_head_ref_and_module, which both matrices already share, so the
classic folders carry byte-identical array formats and the existing
regexes match them. fit_head, compute_all_thresholds, load_resolved_fold
and DEFAULT_THRESHOLD_METHOD come from export_deploy_matrix.py, and the
embedding builders from export_deploy_matrix_classic.py -- so the offline
numbers use IDENTICAL logic to what was baked into each setup's C files.

COST WARNING: this recomputes every embedding from scratch, including the
four slow true-int8 passes (h-width x recurrence). That is the same
situation the selective script is in, but this model has d_inner = 128
against the selective model's 64, and four slow passes instead of two, so
budget roughly four times the wall clock. Use --only while iterating.

Usage:
    python mcu/compute_offline_metrics_classic.py \
        --config 352f70960ed3.yaml --held-out-case 1 --model-hash b77482e85dc3
    # optional: --only w_all_pertensor_qab_euclidean_fp32
"""
import argparse
import json
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
from checks.smoke.quant.true_int8_sim_classic import calibrate_missing_ranges_classic

from mcu.export_deploy_matrix import (
    load_resolved_fold, fit_head, compute_all_thresholds, DEFAULT_THRESHOLD_METHOD,
)
from mcu.compute_offline_metrics import (
    check_ref_parity, score_int8_device_path, write_csv_row,
)
from mcu.export_deploy_matrix_classic import (
    TOTAL_COMBOS, build_classic_setup_matrix,
    build_classic_fake_quant_embeddings, build_classic_true_int8_embeddings,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--deploy-root", default="mcu/deploy")
    parser.add_argument("--only", default=None,
                        help=f"comma-separated setup identifiers (default: all {TOTAL_COMBOS})")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    seed = cfg["seed"]
    if m["selective"]:
        raise ValueError("this matrix handles selective=False only; "
                         "use mcu/compute_offline_metrics.py for selective configs")
    if m["discretization"] != "euler":
        raise ValueError(f"classic backbones assume euler; config says {m['discretization']!r}")

    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dims = (d_model, d_state, d_inner, d_conv, n_layers)

    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    if not (base_dir / "ckpt.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {base_dir/'ckpt.pt'}")

    # Same source as export_deploy_matrix_classic.py and run_deployment_test.py --
    # this identity is what makes an offline/online comparison meaningful.
    fold = load_resolved_fold(base_dir)
    test_rows = fold["test"].reset_index(drop=True)
    test_labels = (test_rows["label"].values == "anomaly").astype(int)
    clip_names = [Path(p).stem for p in test_rows["path"]]
    print(f"Test split: {len(test_rows)} clips "
          f"({int((test_labels == 0).sum())} normal, {int((test_labels == 1).sum())} anomaly)")

    norm_mean, norm_std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    norm_stats = (norm_mean, norm_std)

    setups = build_classic_setup_matrix()
    if args.only:
        wanted = set(args.only.split(","))
        setups = [s for s in setups if s["identifier"] in wanted]
        if not setups:
            raise ValueError(f"--only matched no setups; valid ids: "
                             f"{[s['identifier'] for s in build_classic_setup_matrix()]}")

    # Shared true-int8 calibration: identical across both h-widths AND both
    # recurrence forms, since both read the same ranges.json entries.
    ti_ranges = None
    ti_hook_scales = None
    if any(s["family"] == "true_int8" for s in setups):
        ti_ranges = load_ranges_scales(base_dir)
        model_for_hooks = SSMBackbone(**m)
        ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
        model_for_hooks.load_state_dict(ckpt["model"])
        model_for_hooks.eval()
        print("Calibrating classic true-int8 missing ranges (once, shared)...")
        ti_hook_scales = calibrate_missing_ranges_classic(
            model_for_hooks, fold, norm_mean, norm_std, n_layers)

    emb_cache = {}

    def get_embeddings(setup):
        key = setup["backbone_key"]
        if key in emb_cache:
            return emb_cache[key]
        print(f"  computing embeddings for backbone '{key}'...")
        if setup["family"] == "fake_quant":
            emb = build_classic_fake_quant_embeddings(
                cfg, base_dir, fold, norm_stats, setup["granularity"],
                setup["recurrence"], setup["activation_group"])
        else:
            emb = build_classic_true_int8_embeddings(
                cfg, base_dir, fold, norm_stats, dims, ti_ranges, ti_hook_scales,
                setup["h_width"], setup["recurrence"])
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
    all_identifiers = [s["identifier"] for s in build_classic_setup_matrix()]

    for setup in setups:
        combo_number = all_identifiers.index(setup["identifier"]) + 1
        ident = setup["identifier"]
        setup_dir = setups_root / ident
        if not setup_dir.is_dir():
            print(f"\n[{combo_number:2d}/{TOTAL_COMBOS}] {ident}: SKIP -- not yet exported "
                  f"by export_deploy_matrix_classic.py")
            continue
        setup["_setup_dir"] = setup_dir  # threaded through to check_ref_parity
        print(f"\n[{combo_number:2d}/{TOTAL_COMBOS}] {ident}")

        offline_dir = setup_dir / "offline"
        if offline_dir.exists():
            # Idempotent per run: without this, re-running for a setup
            # already processed would silently APPEND a duplicate row to
            # every per-clip CSV, quietly corrupting exactly the files this
            # exists to keep trustworthy.
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
            "matrix": "classic",
            "config": args.config, "held_out_case": args.held_out_case,
            "model_hash": args.model_hash,
            "selective": False,
            "recurrence": setup["recurrence"],
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

    print("\nDone. Offline artifacts written under each setup's offline/ subfolder.")


if __name__ == "__main__":
    main()