"""
Export a trained SSMBackbone checkpoint to plain C weight arrays for the
Nucleo-H7S3L8 Phase 4 port (05_phase_4_backbone_port.md step 3).

GENERATED OUTPUT: ssm_weights.h and ssm_weights.c. Don't hand-edit either;
rerun this script if the checkpoint, config, or fold changes.

Also exports this fold's per-channel normalization stats (mean/std) --
the model runs on apply_normalization(log_mel, mean, std)
(src/features/baselines.py), not on raw log-mel. Computed the same way
src/eval/parity_vectors.py computes it (compute_normalization_stats over
the training folds), so these arrays match that fold's parity_vectors*.npz
"norm_mean"/"norm_std" exactly, not just approximately.

Usage:
    python mcu/export_ssm_weights.py \
        --checkpoint runs/case1/16662b29beb3/ckpt.pt \
        --config f4cd557b7e3b.yaml \
        --held-out-case 1 \
        --out-dir /path/to/ssm-mcu-asd-deploy/nucleo-h7s3l8-ssm-mamba-asd/Appli/Core/Inc
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config_by_name
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone


def format_c_float(v):
    s = f"{v:.9g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s + "f"

def c_array(name, arr, linkage="static const float"):
    flat = np.asarray(arr, dtype=np.float32).flatten()
    values = ", ".join(format_c_float(v) for v in flat)
    return f"{linkage} {name}[{flat.size}] = {{ {values} }};"

def precomputed_A(A_log_tensor):
    return -np.exp(A_log_tensor.detach().cpu().numpy().astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]

    if not m["selective"]:
        raise ValueError("export script only handles selective=True for now")
    if m["discretization"] != "euler":
        raise ValueError(
            f"export script assumes euler discretization for the C port's "
            f"discretize step; config says {m['discretization']!r} -- update "
            f"ssm_backbone.c's ssm_block_step before exporting a zoh config."
        )

    model = SSMBackbone(**m)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    d_model = m["d_model"]
    d_state = m["d_state"]
    d_inner = m["expand"] * d_model
    d_conv = m["d_conv"]
    n_layers = m["n_layers"]
    dt_rank = max(1, d_model // 16)

    fold = get_fold(args.held_out_case)
    norm_mean, norm_std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = f"""#pragma once

#define SSM_D_MODEL {d_model}
#define SSM_D_STATE {d_state}
#define SSM_D_INNER {d_inner}
#define SSM_D_CONV {d_conv}
#define SSM_DT_RANK {dt_rank}
#define SSM_N_LAYERS {n_layers}

typedef struct {{
    const float *in_proj_w;   /* [2*D_INNER][D_MODEL] */
    const float *conv_w;      /* [D_INNER][D_CONV] */
    const float *conv_b;      /* [D_INNER] */
    const float *x_proj_w;    /* [DT_RANK+2*D_STATE][D_INNER] */
    const float *dt_proj_w;   /* [D_INNER][DT_RANK] */
    const float *dt_proj_b;   /* [D_INNER] */
    const float *A;           /* [D_INNER][D_STATE], precomputed -exp(A_log) */
    const float *D;           /* [D_INNER] */
    const float *out_proj_w;  /* [D_MODEL][D_INNER] */
    const float *norm_w;      /* [D_MODEL] */
}} ssm_block_weights_t;

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const float ssm_final_norm_w[SSM_D_MODEL];

/* Per-channel z-score stats for this fold's training data (held-out case
 * {args.held_out_case}). Apply as (raw - ssm_norm_mean) / ssm_norm_std
 * BEFORE feeding a log-mel frame into the backbone -- see
 * SSMBackbone_NormalizeFrame() in ssm_backbone.h. */
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
"""
    (out_dir / "ssm_weights.h").write_text(header)

    lines = ['#include "ssm_weights.h"', ""]
    entries = []
    for i in range(n_layers):
        p = f"blocks_{i}"
        lines.append(c_array(f"{p}_in_proj_w", sd[f"blocks.{i}.in_proj.weight"]))
        lines.append(c_array(f"{p}_conv_w", sd[f"blocks.{i}.conv.weight"]))
        lines.append(c_array(f"{p}_conv_b", sd[f"blocks.{i}.conv.bias"]))
        lines.append(c_array(f"{p}_x_proj_w", sd[f"blocks.{i}.x_proj.weight"]))
        lines.append(c_array(f"{p}_dt_proj_w", sd[f"blocks.{i}.dt_proj.weight"]))
        lines.append(c_array(f"{p}_dt_proj_b", sd[f"blocks.{i}.dt_proj.bias"]))
        lines.append(c_array(f"{p}_A", precomputed_A(sd[f"blocks.{i}.A_log"])))
        lines.append(c_array(f"{p}_D", sd[f"blocks.{i}.D"]))
        lines.append(c_array(f"{p}_out_proj_w", sd[f"blocks.{i}.out_proj.weight"]))
        lines.append(c_array(f"{p}_norm_w", sd[f"norms.{i}.weight"]))
        lines.append("")
        entries.append(
            f"    {{ {p}_in_proj_w, {p}_conv_w, {p}_conv_b, {p}_x_proj_w, "
            f"{p}_dt_proj_w, {p}_dt_proj_b, {p}_A, {p}_D, {p}_out_proj_w, "
            f"{p}_norm_w }}"
        )

    lines.append(c_array("ssm_final_norm_w", sd["final_norm.weight"], linkage="const float"))
    lines.append("")
    lines.append("const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS] = {")
    lines.append(",\n".join(entries))
    lines.append("};")
    lines.append("")
    lines.append(c_array("ssm_norm_mean", norm_mean, linkage="const float"))
    lines.append(c_array("ssm_norm_std", norm_std, linkage="const float"))

    (out_dir / "ssm_weights.c").write_text("\n".join(lines))
    print(f"Wrote {out_dir/'ssm_weights.h'} and {out_dir/'ssm_weights.c'}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()