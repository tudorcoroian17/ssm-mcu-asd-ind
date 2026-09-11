"""
-----------------------------------------------------------------------------
**ARCHIVED** - Use the export_deploy_matrix.py script to generate full deploy
packages. This version is no longer supported or tested.
-----------------------------------------------------------------------------
Export a trained SSMBackbone checkpoint to the true-int8 arithmetic scheme's
C weights/scales/LUTs -- Phase 5's second deployment target on the
Nucleo-H7S3L8, alongside the existing fake-quant weight_act_boundaries_ranges
scheme.

Unlike export_ssm_weights.py, this does NOT generate SSM_<FIELD> macros --
there is nothing to switch between here (every scheme-specific decision for
this backbone is "which h-width", not "quantized or not"). It directly
mirrors checks/smoke/quant/true_int8_sim.py's own weight/scale/LUT
preparation (load_quantized_weights, prepare_quantized_model, build_lut),
IMPORTED AND CALLED UNCHANGED -- this script only formats what that module
already computed and validated into C. If the exported numbers ever
disagree with a Python-side AUC/parity check, the bug is in this script's
formatting, never in a re-derivation of the math.

GENERATED OUTPUT: ssm_weights.h and ssm_weights.c, plus a copy of the
canonical ssm_backbone.h/.c from BACKBONE_SRC_DIR, so every scheme's deploy
folder is self-contained (same convention export_ssm_weights.py uses for
the fake-quant schemes).

h-WIDTH: baked in as a #define in the generated ssm_weights.h
(SSM_H_WIDTH_INT16, present only for --h-width int16). ssm_backbone.h reads
that flag to choose ssm_h_t's type and SSM_H_CLIP -- the .c source is
identical for both widths; only this generated flag and the numeric value
of each layer's s_h differ.

Also exports this fold's per-channel normalization stats (mean/std) --
identical mechanism to export_ssm_weights.py.

Usage:
    python mcu/export_true_int8_weights.py \
        --checkpoint runs/case1/16662b29beb3/ckpt.pt \
        --config f4cd557b7e3b.yaml --held-out-case 1 \
        --model-hash 16662b29beb3 \
        --h-width int8 --out-dir mcu/deploy/case1/true_int8_h8

    python mcu/export_true_int8_weights.py \
        --checkpoint runs/case1/16662b29beb3/ckpt.pt \
        --config f4cd557b7e3b.yaml --held-out-case 1 \
        --model-hash 16662b29beb3 \
        --h-width int16 --out-dir mcu/deploy/case1/true_int8_h16
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from src.config import load_config_by_name
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone

from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import (
    H_WIDTHS, calibrate_missing_ranges, prepare_quantized_model,
)

# Canonical, hand-maintained true-int8 backbone source. Copied byte-for-byte
# into every --out-dir. Edit these two files directly when the backbone
# needs to change; this script only ever reads and copies them.
BACKBONE_SRC_DIR = Path(__file__).parent / "ssm_true_int8_src"

# Scale dict keys (true_int8_sim.py's scales[i], after stripping the
# 'block{i}.' prefix) -> the ssm_true_int8_layer_t struct field each one
# fills. Order here is cosmetic (dict iteration), but every key must exist
# in scales[i] or this will KeyError loudly rather than silently emit a
# wrong/missing scale.
SCALE_FIELD_MAP = {
    "norm_out": "s_norm_out",
    "conv_out": "s_conv_out",
    "u_post_conv_silu": "s_u_post_conv_silu",
    "z_gate": "s_z_gate",
    "x_proj_delta_low_out": "s_x_proj_delta_low_out",
    "B": "s_B",
    "C": "s_C",
    "dt_proj_out": "s_dt_proj_out",
    "delta": "s_delta",
    "A_bar": "s_A_bar",
    "B_bar": "s_B_bar",
    "h_scale": "s_h",
    "y_scan": "s_y_scan",
    "y_gated": "s_y_gated",
    "block_output": "s_block_output",
}


def format_c_float(v):
    s = f"{v:.9g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s + "f"


def c_int8_array(name, arr):
    flat = np.asarray(arr, dtype=np.int8).flatten()
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const int8_t {name}[{flat.size}] = {{ {values} }};"


def c_float_array(name, arr, linkage="static const float"):
    flat = np.asarray(arr, dtype=np.float32).flatten()
    values = ", ".join(format_c_float(v) for v in flat)
    return f"{linkage} {name}[{flat.size}] = {{ {values} }};"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--h-width", choices=list(H_WIDTHS), required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]

    if not m["selective"]:
        raise ValueError("true-int8 export only handles selective=True for now")
    if m["discretization"] != "euler":
        raise ValueError(
            f"true-int8 backbone assumes euler discretization; config says "
            f"{m['discretization']!r} -- update ssm_true_int8_src/ssm_backbone.c "
            f"before exporting a zoh config."
        )

    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dt_rank = max(1, d_model // 16)

    model = SSMBackbone(**m)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    fold = get_fold(args.held_out_case)
    norm_mean, norm_std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])

    base_dir = Path(args.checkpoint).parent
    ranges = load_ranges_scales(base_dir)

    print("Calibrating missing ranges (dt_proj/conv/norm hooks)...")
    hook_scales = calibrate_missing_ranges(model, fold, norm_mean, norm_std, n_layers, dt_rank)

    print(f"Preparing quantized weights, scales, and LUTs [h={args.h_width}]...")
    weights, scales, luts_per_layer = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, args.h_width)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ssm_weights.h -----------------------------------------------------
    # ssm_true_int8_layer_t is DEFINED here (not in the canonical
    # ssm_backbone.h) so the #include stays one-directional: ssm_backbone.h
    # includes ssm_weights.h for the macros, the h-width flag, this struct
    # type, and the extern data below. ssm_weights.h never includes
    # ssm_backbone.h back -- that would be circular (see ssm_backbone.h's
    # own comment on this).
    h_width_define = "#define SSM_H_WIDTH_INT16\n" if args.h_width == "int16" else ""
    header = f"""#pragma once

#include <stdint.h>

{h_width_define}#define SSM_D_MODEL {d_model}
#define SSM_D_STATE {d_state}
#define SSM_D_INNER {d_inner}
#define SSM_D_CONV {d_conv}
#define SSM_DT_RANK {dt_rank}
#define SSM_N_LAYERS {n_layers}

/* One layer's weights, scales, and LUTs -- true int8 arithmetic scheme.
 * Fixed shape regardless of h-width (only the VALUES of s_h and
 * ssm_backbone.h's ssm_h_t typedef change between the two exports).
 * All 2D weight arrays are flat, row-major const int8_t* -- indexed
 * manually inside ssm_backbone.c as w[row * n_cols + col]. */
typedef struct {{
	/* in_proj, conceptually split into u (first SSM_D_INNER output rows)
	 * and z (last SSM_D_INNER rows) -- both share one scale, since both
	 * came from quantizing the SAME (2*SSM_D_INNER, SSM_D_MODEL) tensor
	 * as a whole. */
	const int8_t *in_proj_u_w_q; /* [SSM_D_INNER][SSM_D_MODEL] */
	const int8_t *in_proj_z_w_q; /* [SSM_D_INNER][SSM_D_MODEL] */
	float in_proj_w_scale;

	const int8_t *conv_w_q;      /* [SSM_D_INNER][SSM_D_CONV] */
	float conv_w_scale;
	const float *conv_b;         /* [SSM_D_INNER], real units -- bias is
	                               * never separately quantized (findings/521:
	                               * biases are small; added in real units at
	                               * each op's rescale point). */

	const int8_t *x_proj_w_q;    /* [SSM_DT_RANK + 2*SSM_D_STATE][SSM_D_INNER] */
	float x_proj_w_scale;

	const int8_t *dt_proj_w_q;   /* [SSM_D_INNER][SSM_DT_RANK] */
	float dt_proj_w_scale;
	const float *dt_proj_b;      /* [SSM_D_INNER], real units */

	const int8_t *out_proj_w_q;  /* [SSM_D_MODEL][SSM_D_INNER] */
	float out_proj_w_scale;

	const float *A;   /* [SSM_D_INNER][SSM_D_STATE], real units --
	                    * precomputed -exp(A_log_q * A_log_scale) once at
	                    * export time. Never re-quantized at runtime. */
	const float *D;   /* [SSM_D_INNER], real units, precomputed D_q*D_scale */
	const float *norm_w; /* [SSM_D_MODEL], RMSNorm weight -- genuinely never
	                       * quantized (true_int8_sim.py's quantized_forward
	                       * calls rmsnorm_fp32 with the raw fp32 state-dict
	                       * weight, not the quantized copy it also builds). */

	/* Calibrated scales, static per layer. Names match true_int8_sim.py's
	 * scales[i] dict keys exactly (see SCALE_FIELD_MAP in this export
	 * script). */
	float s_norm_out;
	float s_conv_out;
	float s_u_post_conv_silu;
	float s_z_gate;
	float s_x_proj_delta_low_out;
	float s_B;
	float s_C;
	float s_dt_proj_out;
	float s_delta;
	float s_A_bar;
	float s_B_bar;
	float s_h;         /* = ranges['h'] * (127/SSM_H_CLIP) -- depends on
	                     * h-width, baked in at export time. */
	float s_y_scan;
	float s_y_gated;
	float s_block_output;

	/* 256-entry LUTs, index i+128 holds f(i) for i in [-128,127]. */
	const int8_t *lut_softplus;  /* s_dt_proj_out domain -> s_delta domain */
	const int8_t *lut_silu_conv; /* s_conv_out domain -> s_u_post_conv_silu domain */
	const int8_t *lut_silu_z;    /* s_z_gate domain -> self-derived domain */
	float s_silu_z;              /* the LUT's own output scale (no independent
	                               * calibration exists for silu-at-z-gate) */
}} ssm_true_int8_layer_t;

/* Deliberately does NOT #include "ssm_backbone.h" -- nothing below this
 * point needs ssm_h_t or SSMBackbone_State, and ssm_backbone.h already
 * #includes THIS file (for the macros, the h-width flag, and this
 * struct type). Including it back here would make the dependency
 * circular for no reason; ssm_weights.c only ever needs this file. */

extern const ssm_true_int8_layer_t ssm_layers[SSM_N_LAYERS];
extern const float ssm_final_norm_w[SSM_D_MODEL];
extern const float ssm_final_norm_scale;
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
"""
    (out_dir / "ssm_weights.h").write_text(header)

    # ---- ssm_weights.c ----------------------------------------------------
    decl_lines = []
    layer_inits = []

    for i in range(n_layers):
        w = weights[i]
        s = scales[i]
        lut = luts_per_layer[i]
        p = f"L{i}_"  # per-layer prefix so array names can't collide across layers

        decl_lines.append(c_int8_array(f"{p}in_proj_u_w_q", w["in_proj_u_q"]))
        decl_lines.append(c_int8_array(f"{p}in_proj_z_w_q", w["in_proj_z_q"]))
        decl_lines.append(c_int8_array(f"{p}conv_w_q", w["conv_w_q"]))
        decl_lines.append(c_float_array(f"{p}conv_b", w["conv_b_fp32"]))
        decl_lines.append(c_int8_array(f"{p}x_proj_w_q", w["x_proj_w_q"]))
        decl_lines.append(c_int8_array(f"{p}dt_proj_w_q", w["dt_proj_w_q"]))
        decl_lines.append(c_float_array(f"{p}dt_proj_b", w["dt_proj_b_fp32"]))
        decl_lines.append(c_int8_array(f"{p}out_proj_w_q", w["out_proj_w_q"]))
        decl_lines.append(c_float_array(f"{p}A", w["A_real"]))
        decl_lines.append(c_float_array(f"{p}D", w["D_real"]))
        # norm_w: the RAW fp32 state-dict weight, NOT w["norm_w_q"]/scale --
        # true_int8_sim.py's quantized_forward calls rmsnorm_fp32 with the
        # unquantized weight; that quantized copy it also builds is unused.
        decl_lines.append(c_float_array(f"{p}norm_w", sd[f"norms.{i}.weight"].numpy()))
        decl_lines.append(c_int8_array(f"{p}lut_softplus", lut["softplus"]))
        decl_lines.append(c_int8_array(f"{p}lut_silu_conv", lut["silu_conv"]))
        decl_lines.append(c_int8_array(f"{p}lut_silu_z", lut["silu_z"]))
        decl_lines.append("")

        scale_inits = ", ".join(
            f".{field} = {format_c_float(s[key])}" for key, field in SCALE_FIELD_MAP.items()
        )

        layer_inits.append(f"""    {{
        .in_proj_u_w_q = {p}in_proj_u_w_q, .in_proj_z_w_q = {p}in_proj_z_w_q,
        .in_proj_w_scale = {format_c_float(w["in_proj_w_scale"])},
        .conv_w_q = {p}conv_w_q, .conv_w_scale = {format_c_float(w["conv_w_scale"])},
        .conv_b = {p}conv_b,
        .x_proj_w_q = {p}x_proj_w_q, .x_proj_w_scale = {format_c_float(w["x_proj_w_scale"])},
        .dt_proj_w_q = {p}dt_proj_w_q, .dt_proj_w_scale = {format_c_float(w["dt_proj_w_scale"])},
        .dt_proj_b = {p}dt_proj_b,
        .out_proj_w_q = {p}out_proj_w_q, .out_proj_w_scale = {format_c_float(w["out_proj_w_scale"])},
        .A = {p}A, .D = {p}D, .norm_w = {p}norm_w,
        {scale_inits},
        .lut_softplus = {p}lut_softplus, .lut_silu_conv = {p}lut_silu_conv,
        .lut_silu_z = {p}lut_silu_z, .s_silu_z = {format_c_float(lut["silu_z_scale"])},
    }}""")

    lines = ['#include "ssm_weights.h"', ""]
    lines.extend(decl_lines)
    lines.append("const ssm_true_int8_layer_t ssm_layers[SSM_N_LAYERS] = {")
    lines.append(",\n".join(layer_inits))
    lines.append("};")
    lines.append("")
    lines.append(c_float_array("ssm_final_norm_w", sd["final_norm.weight"].numpy(), linkage="const float"))
    lines.append(f'const float ssm_final_norm_scale = {format_c_float(ranges["final_norm_output"])};')
    lines.append("")
    lines.append(c_float_array("ssm_norm_mean", norm_mean, linkage="const float"))
    lines.append(c_float_array("ssm_norm_std", norm_std, linkage="const float"))

    (out_dir / "ssm_weights.c").write_text("\n".join(lines) + "\n")

    for fname in ("ssm_backbone.h", "ssm_backbone.c"):
        shutil.copy(BACKBONE_SRC_DIR / fname, out_dir / fname)

    print(f"Wrote {out_dir/'ssm_weights.h'}, {out_dir/'ssm_weights.c'}, "
          f"and copied ssm_backbone.h/.c from {BACKBONE_SRC_DIR}")
    print(f"h-width: {args.h_width}  (SSM_H_CLIP = {H_WIDTHS[args.h_width]})")


if __name__ == "__main__":
    main()