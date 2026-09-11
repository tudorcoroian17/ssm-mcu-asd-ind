"""
Master deployment-matrix exporter for the Nucleo-H7S3L8.

Builds every deployment combo for one (config, held-out case, model hash)
into its own self-contained, unmixable folder:

    mcu/deploy/case<N>/<model_hash>/setups/<setup_identifier>/

Each setup folder contains exactly the eight C files the CubeIDE project
needs (ssm_backbone.h/.c, ssm_weights.h/.c, ssm_head_ref.h/.c,
ssm_distance_head.h/.c) plus a human-readable README.md and a
machine-readable diagnostics.json. Point one CubeIDE source folder at one
setup directory; the folder is the only thing that changes between combos.

WHY THIS SCRIPT EXISTS / WHAT IT REPLACES: the three earlier, single-
purpose exporters (export_ssm_weights.py, export_true_int8_weights.py,
export_reference_heads.py) are archived. Every piece of emission logic they
held is REDEFINED here, so this file has no runtime dependency on them --
only on the shared library primitives (thresholds.py, weight_quant_parity.py,
activation_quant*.py, true_int8_sim.py, backbone.py, folds.py,
auc_pauc.DISTANCE_HEADS) that those exporters also used. That keeps the
archived scripts as a frozen reference without this one drifting from them.

SINGLE HEAD PER FOLDER: each folder scores exactly ONE head (euclidean OR
knn16). The C SSMHeadResult struct still carries all four fields so main.c
is byte-for-byte identical across all 18 folders, but the inactive head's
fields are set to a visible sentinel (score = -1.0, anomaly = 0) on the
wire, so a folder can never be silently confused for its sibling that
scores the other head off the same backbone.

DEFAULT THRESHOLD baked into the C: percentile_same_machine_99.0 (the p99
same-machine percentile). diagnostics.json carries all 14 threshold methods
with their values and resulting test metrics, so any one can be swapped in
by hand by editing a single constant in ssm_head_ref.c (float form for
fp32 heads; both float and int32 sumsq forms for int8 heads) -- the README
lists every method's number next to the swap instructions.

Usage:
    python mcu/export_deploy_matrix.py \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3
    # optional: --only w_all_pertensor_euclidean_fp32,true_int8_h8_knn16_int8
    #           --deploy-root mcu/deploy
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
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
from src.eval.thresholds import (
    secondary_metrics, percentile_threshold, evt_threshold, parametric_threshold,
    kde_threshold, mad_threshold, iqr_threshold, FAR_BUDGETS,
)

from checks.smoke.quant.weight_quant_parity import (
    should_quantize, quantize_dequantize, WEIGHT_MODES, GRANULARITIES,
)
from checks.smoke.quant.activation_quant import ActivationQuantizer, ACTIVATION_GROUPS
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import (
    H_WIDTHS, calibrate_missing_ranges, prepare_quantized_model, quantized_forward,
)

# Canonical, hand-maintained backbone sources -- copied verbatim, never
# generated (same as the archived exporters treated them).
BACKBONE_SRC_FAKE_QUANT = Path(__file__).parent / "ssm_backbone_src"
BACKBONE_SRC_TRUE_INT8 = Path(__file__).parent / "ssm_true_int8_src"

DEFAULT_THRESHOLD_METHOD = "percentile_same_machine_99.0"
KNN16_N_CLUSTERS = 16


# =====================================================================
# SETUP MATRIX -- the 18 combos, as data. Grouped later by backbone so
# each distinct backbone's embeddings are computed exactly once.
# =====================================================================

def build_setup_matrix():
    """Returns the 18 setups. Each is a dict fully describing one folder.
    backbone_key groups setups that share an identical backbone forward
    pass (so embeddings compute once per key, not once per setup)."""
    setups = []

    # Combos 1-8: weight-only int8, no activation quant. Two weight modes x
    # two granularities x two heads.
    for weight_mode, wm_tag in (("all", "all"), ("projections", "proj")):
        for granularity, g_tag in (("per-channel", "perchannel"), ("per-tensor", "pertensor")):
            for head in ("euclidean", "knn16"):
                setups.append({
                    "family": "fake_quant",
                    "weight_mode": weight_mode,
                    "granularity": granularity,
                    "activation_group": "none",
                    "head": head,
                    "head_precision": "fp32",
                    "backbone_key": f"fake_{wm_tag}_{g_tag}_none",
                    "identifier": f"w_{wm_tag}_{g_tag}_{head}_fp32",
                })

    # Combos 9-10: weight all/per-tensor + activation boundaries.
    for head in ("euclidean", "knn16"):
        setups.append({
            "family": "fake_quant",
            "weight_mode": "all",
            "granularity": "per-tensor",
            "activation_group": "boundaries",
            "head": head,
            "head_precision": "fp32",
            "backbone_key": "fake_all_pertensor_boundaries",
            "identifier": f"w_all_pertensor_actboundaries_{head}_fp32",
        })

    # Combos 11-18: true int8, two h-widths x two heads x two head precisions.
    for h_width, h_tag in (("int8", "h8"), ("int16", "h16")):
        for head in ("euclidean", "knn16"):
            for head_precision in ("fp32", "int8"):
                setups.append({
                    "family": "true_int8",
                    "h_width": h_width,
                    "head": head,
                    "head_precision": head_precision,
                    "backbone_key": f"true_int8_{h_tag}",
                    "identifier": f"true_int8_{h_tag}_{head}_{head_precision}",
                })

    return setups


def load_resolved_fold(base_dir):
    manifest_path = base_dir / "embeddings" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No per-model manifest at {manifest_path} -- "
                                f"has this model's evaluation pipeline been run?")
    manifest = pd.read_csv(manifest_path)
    if "used_in" not in manifest.columns:
        raise KeyError(f"{manifest_path} has no 'used_in' column "
                       f"(columns: {list(manifest.columns)})")

    return {
        "train": manifest[manifest["used_in"] == "training"].reset_index(drop=True),
        "test": manifest[manifest["used_in"] == "test"].reset_index(drop=True),
        "calib_normal": manifest[manifest["used_in"] == "machine_calibration"].reset_index(drop=True),
    }


# =====================================================================
# C-EMISSION HELPERS (shared)
# =====================================================================

def format_c_float(v):
    s = f"{v:.9g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s + "f"


def c_array(name, arr, linkage="static const float"):
    flat = np.asarray(arr, dtype=np.float32).flatten()
    values = ", ".join(format_c_float(v) for v in flat)
    return f"{linkage} {name}[{flat.size}] = {{ {values} }};"


def c_int8_array(name, arr, linkage="static const int8_t"):
    flat = np.asarray(arr, dtype=np.int8).flatten()
    values = ", ".join(str(int(v)) for v in flat)
    return f"{linkage} {name}[{flat.size}] = {{ {values} }};"


def c_array_1d_public(name, arr):
    return c_array(name, arr, linkage="const float")


def c_int8_array_1d_public(name, arr):
    flat = np.asarray(arr, dtype=np.int8).flatten()
    values = ", ".join(str(int(v)) for v in flat)
    return f"const int8_t {name}[{flat.size}] = {{ {values} }};"


# =====================================================================
# FAKE-QUANT WEIGHT EMISSION (ssm_weights.h/.c)
# Ported verbatim from the archived export_ssm_weights.py -- the
# positional-struct-init discipline it documents is load-bearing: a
# field's struct-member declaration and its initializer value MUST be
# appended in the same loop iteration, or every field after it silently
# misaligns. Do not "tidy" these two into separate blocks.
# =====================================================================

FQ_ACTIVATION_GROUPS = ("none", "boundaries")
FQ_BOUNDARY_FIELDS_PER_LAYER = {
    "u":         ("block{i}.u_post_conv_silu", "SSM_QUANT_U"),
    "z":         ("block{i}.z_gate",            "SSM_QUANT_Z"),
    "y_gated":   ("block{i}.y_gated",            "SSM_QUANT_Y_GATED"),
    "block_out": ("block{i}.block_output",       "SSM_QUANT_BLOCK_OUT"),
}


def fq_quantize_int8_pertensor(w_np):
    max_abs = float(np.max(np.abs(w_np)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1e-12
    q = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return q, np.array([scale], dtype=np.float32)


def fq_quantize_int8_perchannel(w_np):
    reduce_axes = tuple(range(1, w_np.ndim))
    max_abs = np.max(np.abs(w_np), axis=reduce_axes)
    scale = np.where(max_abs == 0.0, 1e-12, max_abs / 127.0).astype(np.float32)
    scale_bc = scale.reshape((-1,) + (1,) * (w_np.ndim - 1))
    q = np.clip(np.round(w_np / scale_bc), -127, 127).astype(np.int8)
    return q, scale


def fq_emit_2d_field(field_name, macro_name, tensor, quantize, granularity, ncols_macro,
                     decl_lines, member_decls, macro_lines):
    flat = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    bare = field_name.split('_', 2)[-1]
    if not quantize:
        decl_lines.append(c_array(field_name, flat))
        member_decls.append(f"    const float *{bare};")
        macro_lines.append(f"#define {macro_name}(w, r, c) ((w)->{bare}[(r) * {ncols_macro} + (c)])")
        return field_name
    use_perchannel = granularity == "per-channel" and flat.ndim >= 2
    q, scale = (fq_quantize_int8_perchannel(flat) if use_perchannel else fq_quantize_int8_pertensor(flat))
    decl_lines.append(c_int8_array(f"{field_name}_q", q))
    decl_lines.append(c_array(f"{field_name}_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_q;")
    member_decls.append(f"    const float *{bare}_scale;")
    scale_idx = "(r)" if use_perchannel else "0"
    macro_lines.append(
        f"#define {macro_name}(w, r, c) "
        f"((float)(w)->{bare}_q[(r) * {ncols_macro} + (c)] * (w)->{bare}_scale[{scale_idx}])")
    return f"{field_name}_q, {field_name}_scale"


def fq_emit_1d_field(field_name, macro_name, tensor, quantize, decl_lines, member_decls, macro_lines):
    flat = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    bare = field_name.split('_', 2)[-1]
    if not quantize:
        decl_lines.append(c_array(field_name, flat))
        member_decls.append(f"    const float *{bare};")
        macro_lines.append(f"#define {macro_name}(w, i) ((w)->{bare}[(i)])")
        return field_name
    q, scale = fq_quantize_int8_pertensor(flat)
    decl_lines.append(c_int8_array(f"{field_name}_q", q))
    decl_lines.append(c_array(f"{field_name}_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_q;")
    member_decls.append(f"    const float *{bare}_scale;")
    macro_lines.append(f"#define {macro_name}(w, i) ((float)(w)->{bare}_q[(i)] * (w)->{bare}_scale[0])")
    return f"{field_name}_q, {field_name}_scale"


def fq_emit_A_field(field_name, macro_name, A_log_tensor, quantize, granularity,
                    decl_lines, member_decls, macro_lines):
    flat = np.asarray(A_log_tensor.detach().cpu().numpy(), dtype=np.float32)
    bare = field_name.split('_', 2)[-1]
    if not quantize:
        A = -np.exp(flat)
        decl_lines.append(c_array(field_name, A))
        member_decls.append(f"    const float *{bare};")
        macro_lines.append(f"#define {macro_name}(w, r, c) ((w)->{bare}[(r) * SSM_D_STATE + (c)])")
        return field_name
    use_perchannel = granularity == "per-channel" and flat.ndim >= 2
    q, scale = (fq_quantize_int8_perchannel(flat) if use_perchannel else fq_quantize_int8_pertensor(flat))
    decl_lines.append(c_int8_array(f"{field_name}_log_q", q))
    decl_lines.append(c_array(f"{field_name}_log_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_log_q;")
    member_decls.append(f"    const float *{bare}_log_scale;")
    scale_idx = "(r)" if use_perchannel else "0"
    macro_lines.append(
        f"#define {macro_name}(w, r, c) "
        f"(-expf((float)(w)->{bare}_log_q[(r) * SSM_D_STATE + (c)] * (w)->{bare}_log_scale[{scale_idx}]))")
    return f"{field_name}_log_q, {field_name}_log_scale"


def emit_fake_quant_weights(out_dir, cfg, sd, dims, norm_stats, weight_mode, granularity,
                            activation_group, checkpoint_dir, held_out_case):
    d_model, d_state, d_inner, d_conv, n_layers, dt_rank = dims
    norm_mean, norm_std = norm_stats

    def is_quantized(sd_key):
        return should_quantize(sd_key, weight_mode)

    act_quantized = activation_group == "boundaries"
    act_scales = {}
    if act_quantized:
        act_scales = load_ranges_scales(checkpoint_dir)

    norm_quantized = is_quantized("norms.0.weight")
    norm_member_decls = []
    if not norm_quantized:
        norm_member_decls.append("    const float *w;")
        norm_macro = "#define SSM_NORM_W(nw, i) ((nw)->w[(i)])"
    else:
        norm_member_decls.append("    const int8_t *w_q;")
        norm_member_decls.append("    const float *w_scale;")
        norm_macro = "#define SSM_NORM_W(nw, i) ((float)(nw)->w_q[(i)] * (nw)->w_scale[0])"

    decl_lines = []

    def emit_norm_instance(field_name, tensor):
        flat = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        if not norm_quantized:
            decl_lines.append(c_array(field_name, flat))
            return f"{{ {field_name} }}"
        q, scale = fq_quantize_int8_pertensor(flat)
        decl_lines.append(c_int8_array(f"{field_name}_q", q))
        decl_lines.append(c_array(f"{field_name}_scale", scale))
        return f"{{ {field_name}_q, {field_name}_scale }}"

    member_decls = []
    macro_lines = []
    block_inits = []

    fields_2d = [
        ("in_proj_w", "SSM_IN_PROJ_W", lambda i: sd[f"blocks.{i}.in_proj.weight"], lambda i: f"blocks.{i}.in_proj.weight", "SSM_D_MODEL"),
        ("conv_w", "SSM_CONV_W", lambda i: sd[f"blocks.{i}.conv.weight"], lambda i: f"blocks.{i}.conv.weight", "SSM_D_CONV"),
        ("x_proj_w", "SSM_X_PROJ_W", lambda i: sd[f"blocks.{i}.x_proj.weight"], lambda i: f"blocks.{i}.x_proj.weight", "SSM_D_INNER"),
        ("dt_proj_w", "SSM_DT_PROJ_W", lambda i: sd[f"blocks.{i}.dt_proj.weight"], lambda i: f"blocks.{i}.dt_proj.weight", "SSM_DT_RANK"),
        ("out_proj_w", "SSM_OUT_PROJ_W", lambda i: sd[f"blocks.{i}.out_proj.weight"], lambda i: f"blocks.{i}.out_proj.weight", "SSM_D_INNER"),
    ]
    fields_1d = [
        ("conv_b", "SSM_CONV_B", lambda i: sd[f"blocks.{i}.conv.bias"], lambda i: f"blocks.{i}.conv.bias"),
        ("dt_proj_b", "SSM_DT_PROJ_B", lambda i: sd[f"blocks.{i}.dt_proj.bias"], lambda i: f"blocks.{i}.dt_proj.bias"),
        ("D", "SSM_D_PARAM", lambda i: sd[f"blocks.{i}.D"], lambda i: f"blocks.{i}.D"),
    ]

    for i in range(n_layers):
        init_parts = []
        shape_lists = (member_decls, macro_lines) if i == 0 else ([], [])
        for field_name, macro_name, tensor_fn, key_fn, ncols in fields_2d:
            init_parts.append(fq_emit_2d_field(
                f"blocks_{i}_{field_name}", macro_name, tensor_fn(i), is_quantized(key_fn(i)),
                granularity, ncols, decl_lines, *shape_lists))
        for field_name, macro_name, tensor_fn, key_fn in fields_1d:
            init_parts.append(fq_emit_1d_field(
                f"blocks_{i}_{field_name}", macro_name, tensor_fn(i), is_quantized(key_fn(i)),
                decl_lines, *shape_lists))
        init_parts.append(fq_emit_A_field(
            f"blocks_{i}_A", "SSM_A", sd[f"blocks.{i}.A_log"], is_quantized(f"blocks.{i}.A_log"),
            granularity, decl_lines, *shape_lists))
        if act_quantized:
            for name in FQ_BOUNDARY_FIELDS_PER_LAYER:
                shape_lists[0].append(f"    float {name}_scale;")
            for name, (key_fmt, _) in FQ_BOUNDARY_FIELDS_PER_LAYER.items():
                init_parts.append(format_c_float(act_scales[key_fmt.format(i=i)]))
        init_parts.append(emit_norm_instance(f"blocks_{i}_norm_w", sd[f"norms.{i}.weight"]))
        decl_lines.append("")
        block_inits.append(f"    {{ {', '.join(init_parts)} }}")

    if act_quantized:
        for name, (_, macro_name) in FQ_BOUNDARY_FIELDS_PER_LAYER.items():
            macro_lines.append(f"#define {macro_name}(w, val) (ssm_quant_dequant((val), (w)->{name}_scale))")
    else:
        for name, (_, macro_name) in FQ_BOUNDARY_FIELDS_PER_LAYER.items():
            macro_lines.append(f"#define {macro_name}(w, val) (val)")

    final_norm_init = emit_norm_instance("ssm_final_norm_w_data", sd["final_norm.weight"])

    final_norm_scale_decl = ""
    final_norm_scale_extern = ""
    if act_quantized:
        final_norm_scale_decl = f"const float ssm_final_norm_scale = {format_c_float(act_scales['final_norm_output'])};"
        final_norm_scale_extern = "extern const float ssm_final_norm_scale;"
        macro_lines.append("#define SSM_QUANT_FINAL_NORM(val) (ssm_quant_dequant((val), ssm_final_norm_scale))")
    else:
        macro_lines.append("#define SSM_QUANT_FINAL_NORM(val) (val)")

    header = f"""#pragma once

#include <stdint.h>
#include <math.h>

#define SSM_D_MODEL {d_model}
#define SSM_D_STATE {d_state}
#define SSM_D_INNER {d_inner}
#define SSM_D_CONV {d_conv}
#define SSM_DT_RANK {dt_rank}
#define SSM_N_LAYERS {n_layers}

typedef struct {{
{chr(10).join(norm_member_decls)}
}} ssm_norm_weights_t;

typedef struct {{
{chr(10).join(member_decls)}
    ssm_norm_weights_t norm_w;
}} ssm_block_weights_t;

{chr(10).join(macro_lines)}
{norm_macro}

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const ssm_norm_weights_t ssm_final_norm_w;
{final_norm_scale_extern}

extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
"""
    (out_dir / "ssm_weights.h").write_text(header)

    lines = ['#include "ssm_weights.h"', ""]
    lines.extend(decl_lines)
    lines.append("const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS] = {")
    lines.append(",\n".join(block_inits))
    lines.append("};")
    lines.append("")
    lines.append(f"const ssm_norm_weights_t ssm_final_norm_w = {final_norm_init};")
    if final_norm_scale_decl:
        lines.append(final_norm_scale_decl)
    lines.append("")
    lines.append(c_array("ssm_norm_mean", norm_mean, linkage="const float"))
    lines.append(c_array("ssm_norm_std", norm_std, linkage="const float"))
    (out_dir / "ssm_weights.c").write_text("\n".join(lines) + "\n")

    for fname in ("ssm_backbone.h", "ssm_backbone.c"):
        shutil.copy(BACKBONE_SRC_FAKE_QUANT / fname, out_dir / fname)


# =====================================================================
# TRUE-INT8 WEIGHT EMISSION (ssm_weights.h/.c)
# Ported from the archived export_true_int8_weights.py. Emits the layer
# struct into the generated header (one-directional include: ssm_backbone.h
# includes this header, never the reverse). h-width is a #define here; the
# canonical ssm_true_int8_src/ssm_backbone.c reads it to pick ssm_h_t.
# =====================================================================

TI_SCALE_FIELD_MAP = {
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


def ti_c_int8_array(name, arr):
    flat = np.asarray(arr, dtype=np.int8).flatten()
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const int8_t {name}[{flat.size}] = {{ {values} }};"


def ti_c_float_array(name, arr, linkage="static const float"):
    flat = np.asarray(arr, dtype=np.float32).flatten()
    values = ", ".join(format_c_float(v) for v in flat)
    return f"{linkage} {name}[{flat.size}] = {{ {values} }};"


def emit_true_int8_weights(out_dir, cfg, sd, dims, norm_stats, ranges, hook_scales, h_width):
    d_model, d_state, d_inner, d_conv, n_layers, dt_rank = dims
    norm_mean, norm_std = norm_stats

    weights, scales, luts_per_layer = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, h_width)

    h_width_define = "#define SSM_H_WIDTH_INT16\n" if h_width == "int16" else ""
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
 * ssm_backbone.h's ssm_h_t typedef change between the two exports). */
typedef struct {{
	const int8_t *in_proj_u_w_q;
	const int8_t *in_proj_z_w_q;
	float in_proj_w_scale;

	const int8_t *conv_w_q;
	float conv_w_scale;
	const float *conv_b;

	const int8_t *x_proj_w_q;
	float x_proj_w_scale;

	const int8_t *dt_proj_w_q;
	float dt_proj_w_scale;
	const float *dt_proj_b;

	const int8_t *out_proj_w_q;
	float out_proj_w_scale;

	const float *A;
	const float *D;
	const float *norm_w;

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
	float s_h;
	float s_y_scan;
	float s_y_gated;
	float s_block_output;

	const int8_t *lut_softplus;
	const int8_t *lut_silu_conv;
	const int8_t *lut_silu_z;
	float s_silu_z;
}} ssm_true_int8_layer_t;

/* Deliberately does NOT #include "ssm_backbone.h" -- ssm_backbone.h
 * includes THIS file for the macros, h-width flag, and this struct type;
 * including it back would be circular. ssm_weights.c needs only this. */

extern const ssm_true_int8_layer_t ssm_layers[SSM_N_LAYERS];
extern const float ssm_final_norm_w[SSM_D_MODEL];
extern const float ssm_final_norm_scale;
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
"""
    (out_dir / "ssm_weights.h").write_text(header)

    decl_lines = []
    layer_inits = []
    for i in range(n_layers):
        w = weights[i]
        s = scales[i]
        lut = luts_per_layer[i]
        p = f"L{i}_"
        decl_lines.append(ti_c_int8_array(f"{p}in_proj_u_w_q", w["in_proj_u_q"]))
        decl_lines.append(ti_c_int8_array(f"{p}in_proj_z_w_q", w["in_proj_z_q"]))
        decl_lines.append(ti_c_int8_array(f"{p}conv_w_q", w["conv_w_q"]))
        decl_lines.append(ti_c_float_array(f"{p}conv_b", w["conv_b_fp32"]))
        decl_lines.append(ti_c_int8_array(f"{p}x_proj_w_q", w["x_proj_w_q"]))
        decl_lines.append(ti_c_int8_array(f"{p}dt_proj_w_q", w["dt_proj_w_q"]))
        decl_lines.append(ti_c_float_array(f"{p}dt_proj_b", w["dt_proj_b_fp32"]))
        decl_lines.append(ti_c_int8_array(f"{p}out_proj_w_q", w["out_proj_w_q"]))
        decl_lines.append(ti_c_float_array(f"{p}A", w["A_real"]))
        decl_lines.append(ti_c_float_array(f"{p}D", w["D_real"]))
        decl_lines.append(ti_c_float_array(f"{p}norm_w", sd[f"norms.{i}.weight"].numpy()))
        decl_lines.append(ti_c_int8_array(f"{p}lut_softplus", lut["softplus"]))
        decl_lines.append(ti_c_int8_array(f"{p}lut_silu_conv", lut["silu_conv"]))
        decl_lines.append(ti_c_int8_array(f"{p}lut_silu_z", lut["silu_z"]))
        decl_lines.append("")
        scale_inits = ", ".join(
            f".{field} = {format_c_float(s[key])}" for key, field in TI_SCALE_FIELD_MAP.items())
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
    lines.append(ti_c_float_array("ssm_final_norm_w", sd["final_norm.weight"].numpy(), linkage="const float"))
    lines.append(f'const float ssm_final_norm_scale = {format_c_float(ranges["final_norm_output"])};')
    lines.append("")
    lines.append(ti_c_float_array("ssm_norm_mean", norm_mean, linkage="const float"))
    lines.append(ti_c_float_array("ssm_norm_std", norm_std, linkage="const float"))
    (out_dir / "ssm_weights.c").write_text("\n".join(lines) + "\n")

    for fname in ("ssm_backbone.h", "ssm_backbone.c"):
        shutil.copy(BACKBONE_SRC_TRUE_INT8 / fname, out_dir / fname)

    return ranges["final_norm_output"]


# =====================================================================
# HEAD EMISSION (ssm_head_ref.h/.c + ssm_distance_head.h/.c)
# Single head per folder. SSMHeadResult keeps all four fields so main.c is
# identical everywhere; the inactive head is a visible sentinel on the wire.
# =====================================================================

# --- fp32 head (weight-only, activation, and true_int8 fp32-head combos) ---

FP32_HEAD_H = """#pragma once

#include <stdint.h>
#include "ssm_weights.h"

/* SINGLE-HEAD build: this folder scores {HEAD_NAME} only. The other head's
 * fields are set to a sentinel (score -1.0, anomaly 0) so this folder can
 * never be confused on the wire for its sibling that scores the other head
 * off the same backbone. SSMHeadResult keeps all four fields regardless, so
 * main.c is byte-for-byte identical across every setup. */

#define SSM_KNN16_N_CLUSTERS 16

typedef struct {
	float euclidean_score;
	float knn16_score;
	uint8_t euclidean_anomaly;
	uint8_t knn16_anomaly;
} SSMHeadResult;

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result);
"""

FP32_HEAD_C_EUCLIDEAN = """#include "ssm_distance_head.h"

#include <math.h>
#include "ssm_head_ref.h"

static float ssm_l2_distance(const float *a, const float *b, int n) {
	float acc = 0.0f;
	for (int i = 0; i < n; i++) {
		float d = a[i] - b[i];
		acc += d * d;
	}
	return sqrtf(acc);
}

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	result->euclidean_score = ssm_l2_distance(embedding, ssm_ref_centroid, SSM_D_MODEL);
	result->euclidean_anomaly = (result->euclidean_score > ssm_threshold_euclidean) ? 1 : 0;

	/* knn16 not scored in this folder -- sentinel. */
	result->knn16_score = -1.0f;
	result->knn16_anomaly = 0;
}
"""

FP32_HEAD_C_KNN16 = """#include "ssm_distance_head.h"

#include <math.h>
#include "ssm_head_ref.h"

static float ssm_l2_distance(const float *a, const float *b, int n) {
	float acc = 0.0f;
	for (int i = 0; i < n; i++) {
		float d = a[i] - b[i];
		acc += d * d;
	}
	return sqrtf(acc);
}

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	float best = INFINITY;
	for (int k = 0; k < SSM_KNN16_N_CLUSTERS; k++) {
		float d = ssm_l2_distance(embedding, ssm_ref_clusters[k], SSM_D_MODEL);
		if (d < best) {
			best = d;
		}
	}
	result->knn16_score = best;
	result->knn16_anomaly = (result->knn16_score > ssm_threshold_knn16) ? 1 : 0;

	/* euclidean not scored in this folder -- sentinel. */
	result->euclidean_score = -1.0f;
	result->euclidean_anomaly = 0;
}
"""

# --- int8 head (true_int8 int8-head combos only) ---

INT8_HEAD_H = """#pragma once

#include <stdint.h>
#include "ssm_weights.h"

/* SINGLE-HEAD, pure-int32 build: this folder scores {HEAD_NAME} only, with
 * the entire decision path in int32 (only the reported score uses a
 * cosmetic sqrt). The other head's fields are a sentinel. SSMHeadResult
 * keeps all four fields so main.c is identical across every setup. Valid
 * ONLY on a true-int8 backbone whose pooled embedding is an exact int8
 * round trip at ssm_final_norm_scale. */

#define SSM_KNN16_N_CLUSTERS 16

typedef struct {
	float euclidean_score;
	float knn16_score;
	uint8_t euclidean_anomaly;
	uint8_t knn16_anomaly;
	int32_t euclidean_sumsq;
	int32_t knn16_sumsq;
} SSMHeadResult;

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result);
"""

INT8_HEAD_C_HEADER = """#include "ssm_distance_head.h"

#include <math.h>
#include "ssm_weights.h"
#include "ssm_head_ref.h"

static int8_t ssm_head_quantize(float real) {
	float q = roundf(real / ssm_final_norm_scale);
	if (q > 127.0f) {
		q = 127.0f;
	}
	if (q < -127.0f) {
		q = -127.0f;
	}
	return (int8_t) q;
}

static int32_t ssm_sumsq(const int8_t *a, const int8_t *b, int n) {
	int32_t acc = 0;
	for (int i = 0; i < n; i++) {
		int32_t d = (int32_t) a[i] - (int32_t) b[i];
		acc += d * d;
	}
	return acc;
}
"""

INT8_HEAD_C_EUCLIDEAN = INT8_HEAD_C_HEADER + """
void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	int8_t embedding_q[SSM_D_MODEL];
	for (int i = 0; i < SSM_D_MODEL; i++) {
		embedding_q[i] = ssm_head_quantize(embedding[i]);
	}

	result->euclidean_sumsq = ssm_sumsq(embedding_q, ssm_ref_centroid_q, SSM_D_MODEL);
	result->euclidean_score = sqrtf((float) result->euclidean_sumsq) * ssm_final_norm_scale;
	result->euclidean_anomaly = (result->euclidean_sumsq > ssm_threshold_euclidean_sumsq) ? 1 : 0;

	/* knn16 not scored in this folder -- sentinel. */
	result->knn16_sumsq = -1;
	result->knn16_score = -1.0f;
	result->knn16_anomaly = 0;
}
"""

INT8_HEAD_C_KNN16 = INT8_HEAD_C_HEADER + """
void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	int8_t embedding_q[SSM_D_MODEL];
	for (int i = 0; i < SSM_D_MODEL; i++) {
		embedding_q[i] = ssm_head_quantize(embedding[i]);
	}

	int32_t best = -1;
	for (int k = 0; k < SSM_KNN16_N_CLUSTERS; k++) {
		int32_t sumsq = ssm_sumsq(embedding_q, ssm_ref_clusters_q[k], SSM_D_MODEL);
		if (best < 0 || sumsq < best) {
			best = sumsq;
		}
	}
	result->knn16_sumsq = best;
	result->knn16_score = sqrtf((float) result->knn16_sumsq) * ssm_final_norm_scale;
	result->knn16_anomaly = (result->knn16_sumsq > ssm_threshold_knn16_sumsq) ? 1 : 0;

	/* euclidean not scored in this folder -- sentinel. */
	result->euclidean_sumsq = -1;
	result->euclidean_score = -1.0f;
	result->euclidean_anomaly = 0;
}
"""


def emit_head_ref_and_module(out_dir, head, head_precision, ref, chosen_threshold, final_norm_scale):
    """Writes ssm_head_ref.h/.c and ssm_distance_head.h/.c for ONE head.
    ref holds centroid + clusters (fitted on this backbone's train embeddings).
    chosen_threshold is the single float baked in as the active default."""
    head_name = "euclidean" if head == "euclidean" else "knn_clustered_16"

    if head_precision == "fp32":
        head_h = FP32_HEAD_H.replace("{HEAD_NAME}", head_name)
        (out_dir / "ssm_distance_head.h").write_text(head_h)
        head_c = FP32_HEAD_C_EUCLIDEAN if head == "euclidean" else FP32_HEAD_C_KNN16
        (out_dir / "ssm_distance_head.c").write_text(head_c)

        # ssm_head_ref: both reference arrays are always present (cheap, and
        # keeps the header uniform); only the active head's threshold is
        # consulted by the emitted ssm_distance_head.c. Inactive threshold is
        # still written so a hand-swap to the other head needs no new export.
        ref_header = """#pragma once

#include "ssm_weights.h"

#define SSM_KNN16_N_CLUSTERS 16

extern const float ssm_ref_centroid[SSM_D_MODEL];
extern const float ssm_ref_clusters[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL];
extern const float ssm_threshold_euclidean;
extern const float ssm_threshold_knn16;
"""
        (out_dir / "ssm_head_ref.h").write_text(ref_header)

        thr_e = chosen_threshold if head == "euclidean" else -1.0
        thr_k = chosen_threshold if head == "knn16" else -1.0
        rows = ",\n    ".join(
            "{ " + ", ".join(format_c_float(v) for v in row) + " }" for row in ref["clusters"])
        body = [
            '#include "ssm_head_ref.h"', "",
            c_array_1d_public("ssm_ref_centroid", ref["centroid"]), "",
            f"const float ssm_ref_clusters[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL] = {{\n    {rows}\n}};", "",
            f"const float ssm_threshold_euclidean = {format_c_float(thr_e)};",
            f"const float ssm_threshold_knn16 = {format_c_float(thr_k)};",
        ]
        (out_dir / "ssm_head_ref.c").write_text("\n".join(body) + "\n")

    else:  # int8 head
        head_h = INT8_HEAD_H.replace("{HEAD_NAME}", head_name)
        (out_dir / "ssm_distance_head.h").write_text(head_h)
        head_c = INT8_HEAD_C_EUCLIDEAN if head == "euclidean" else INT8_HEAD_C_KNN16
        (out_dir / "ssm_distance_head.c").write_text(head_c)

        centroid_q = quantize_ref_int8(ref["centroid"], final_norm_scale)
        clusters_q = quantize_ref_int8(ref["clusters"], final_norm_scale)

        thr_e_sumsq = int(round((chosen_threshold / final_norm_scale) ** 2)) if head == "euclidean" else -1
        thr_k_sumsq = int(round((chosen_threshold / final_norm_scale) ** 2)) if head == "knn16" else -1

        ref_header = """#pragma once

#include <stdint.h>
#include "ssm_weights.h"

#define SSM_KNN16_N_CLUSTERS 16

extern const int8_t ssm_ref_centroid_q[SSM_D_MODEL];
extern const int8_t ssm_ref_clusters_q[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL];
extern const int32_t ssm_threshold_euclidean_sumsq;
extern const int32_t ssm_threshold_knn16_sumsq;
"""
        (out_dir / "ssm_head_ref.h").write_text(ref_header)

        rows = ",\n    ".join(
            "{ " + ", ".join(str(int(v)) for v in row) + " }" for row in clusters_q)
        body = [
            '#include "ssm_head_ref.h"', "",
            c_int8_array_1d_public("ssm_ref_centroid_q", centroid_q), "",
            f"const int8_t ssm_ref_clusters_q[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL] = {{\n    {rows}\n}};", "",
            f"const int32_t ssm_threshold_euclidean_sumsq = {thr_e_sumsq};",
            f"const int32_t ssm_threshold_knn16_sumsq = {thr_k_sumsq};",
        ]
        (out_dir / "ssm_head_ref.c").write_text("\n".join(body) + "\n")


def quantize_ref_int8(real_arr, scale):
    """Quantize a reference vector to int8 against a GIVEN scale
    (ssm_final_norm_scale -- the same scale the embedding lives on, never
    independently re-derived). Warns if any entry would clip."""
    scaled = np.asarray(real_arr, dtype=np.float64) / scale
    n_clipped = int(np.sum(np.abs(scaled) > 127.0))
    if n_clipped > 0:
        print(f"      WARNING: {n_clipped} reference entr"
              f"{'y' if n_clipped == 1 else 'ies'} clip at int8 export "
              f"(magnitude exceeded the embedding's int8 range).")
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)


# =====================================================================
# EMBEDDINGS -- computed once per distinct backbone config, cached by
# backbone_key. Fake-quant path is batched and fast; true-int8 path runs
# the per-frame simulator and is slow (reused from true_int8_auc.py's
# cost profile).
# =====================================================================

def build_fake_quant_embeddings(cfg, base_dir, fold, norm_stats, weight_mode,
                                granularity, activation_group):
    device = "cpu"
    mean, std = norm_stats
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
    model.pooling = "mean"

    quantizer = None
    if activation_group != "none":
        act_scales = load_ranges_scales(base_dir)
        quantizer = ActivationQuantizer(
            group=activation_group, granularity="per-tensor",
            scale_source="ranges", scales=act_scales)

    def embed(split, batch_size=128):
        X = load_fold_clips(fold[split], mean, std)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
                emb = model(batch, mode="pooled", quantizer=quantizer)
                out.append(emb.cpu().numpy())
        return np.concatenate(out, axis=0)

    return {"train": embed("train"), "test": embed("test"), "calib_normal": embed("calib_normal")}


def build_true_int8_embeddings(cfg, base_dir, fold, norm_stats, dims, ranges, hook_scales, h_width):
    d_model, d_state, d_inner, d_conv, n_layers, dt_rank = dims
    mean, std = norm_stats
    model = SSMBackbone(**cfg["model"])
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    weights, scales, luts_per_layer = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, h_width)

    def embed(split):
        rows = fold[split]
        out = np.zeros((len(rows), d_model))
        for idx, (_, row) in enumerate(rows.iterrows()):
            log_mel = np.load(row["cache_path"]).astype(np.float32)
            x = apply_normalization(log_mel, mean, std).astype(np.float32)
            out[idx] = quantized_forward(x, sd, weights, scales, luts_per_layer, ranges,
                                         d_model, d_inner, d_state, d_conv, n_layers, h_width)
            if (idx + 1) % 50 == 0:
                print(f"      {split}: {idx + 1}/{len(rows)}")
        return out

    print(f"    true_int8 [h={h_width}] embeddings (slow, per-frame simulator)...")
    return {"train": embed("train"), "test": embed("test"), "calib_normal": embed("calib_normal")}


# =====================================================================
# HEAD FITTING + ALL 14 THRESHOLD METHODS
# =====================================================================

def fit_head(head, train_emb, seed):
    """Returns (ref dict, scorer). euclidean's centroid is train_emb.mean(0)
    exactly as auc_pauc.score_euclidean computes it -- computed directly here
    rather than round-tripped through the scorer, since it's unambiguous.
    knn16 refits KMeans with identical params to auc_pauc.score_knn_clustered
    (n_clusters=16, n_init=10, random_state=seed, k=1 nearest) to recover the
    cluster centers the C side needs."""
    n_dim = train_emb.shape[1]
    if head == "euclidean":
        centroid = train_emb.mean(axis=0)

        def scorer(emb):
            return np.linalg.norm(emb - centroid, axis=1)
        ref = {"centroid": centroid,
               "clusters": np.zeros((KNN16_N_CLUSTERS, n_dim), dtype=np.float32)}
        return ref, scorer
    else:
        kmeans = KMeans(n_clusters=KNN16_N_CLUSTERS, n_init=10, random_state=seed).fit(train_emb)
        clusters = kmeans.cluster_centers_

        def scorer(emb):
            return np.sort(cdist(emb, clusters), axis=1)[:, 0]
        ref = {"centroid": np.zeros(n_dim, dtype=np.float32), "clusters": clusters}
        return ref, scorer


def compute_all_thresholds(calib_scores, seed):
    """The 14 same-machine threshold methods from thresholds.py's on_held_out
    branch. Returns an ordered dict: method_name -> (threshold, info)."""
    methods = {}
    for pct in (95.0, 99.0, 99.5):
        methods[f"percentile_same_machine_{pct}"] = (
            float(percentile_threshold(calib_scores, pct)),
            {"percentile": pct})
    for far_name, far in FAR_BUDGETS.items():
        methods[f"evt_{far_name}"] = evt_threshold(calib_scores, far)
        methods[f"parametric_{far_name}"] = parametric_threshold(calib_scores, far)
        methods[f"kde_{far_name}"] = kde_threshold(calib_scores, far, seed=seed)
    methods["mad"] = mad_threshold(calib_scores)
    methods["iqr"] = iqr_threshold(calib_scores)
    return methods


# =====================================================================
# README GENERATION
# =====================================================================

def describe_setup(setup):
    """One-line and multi-line human description of a setup's matchup."""
    head_name = "euclidean" if setup["head"] == "euclidean" else "knn_clustered_16"
    if setup["family"] == "fake_quant":
        wm = setup["weight_mode"]
        gran = setup["granularity"]
        act = setup["activation_group"]
        backbone_desc = (f"weight-only int8 ({wm} tensors, {gran})" if act == "none"
                         else f"weight int8 ({wm}, {gran}) + activation-boundaries int8")
        head_desc = f"{head_name} head (fp32 arithmetic)"
    else:
        backbone_desc = f"true int8 arithmetic, h storage = {setup['h_width']}"
        head_prec = "int32 arithmetic" if setup["head_precision"] == "int8" else "fp32 arithmetic"
        head_desc = f"{head_name} head ({head_prec})"
    return backbone_desc, head_desc


def render_readme(setup, combo_number, cfg_name, held_out_case, model_hash,
                  metrics_by_threshold, chosen_method, head, head_precision,
                  auc, pauc, final_norm_scale):
    backbone_desc, head_desc = describe_setup(setup)
    head_name = "euclidean" if head == "euclidean" else "knn_clustered_16"
    is_int8_head = head_precision == "int8"

    lines = []
    lines.append(f"# Deployment setup: `{setup['identifier']}`")
    lines.append("")
    lines.append(f"Combo {combo_number} of the case {held_out_case} deployment matrix.")
    lines.append("")
    lines.append("## Matchup")
    lines.append("")
    lines.append(f"- Config: `{cfg_name}`")
    lines.append(f"- Held-out case: {held_out_case}")
    lines.append(f"- Model hash: `{model_hash}`")
    lines.append(f"- Backbone: {backbone_desc}")
    lines.append(f"- Head: {head_desc}")
    lines.append(f"- Threshold method: percentile (same-machine calibration)")
    lines.append("")
    lines.append("## Files in this folder")
    lines.append("")
    lines.append("Point one STM32CubeIDE source folder at this directory. The eight "
                 "C files are self-contained; no file outside this folder is read.")
    lines.append("")
    lines.append("| File | Role |")
    lines.append("| :--- | :--- |")
    lines.append("| `ssm_backbone.h` / `.c` | Canonical backbone (copied, not generated). |")
    lines.append("| `ssm_weights.h` / `.c` | This setup's quantized weights, scales" +
                 (", LUTs." if setup["family"] == "true_int8" else "."))
    lines.append("| `ssm_head_ref.h` / `.c` | This setup's reference vectors + baked-in threshold. |")
    lines.append("| `ssm_distance_head.h` / `.c` | Single-head scorer (this folder: "
                 f"{head_name} only). |")
    lines.append("")
    lines.append("The sibling head is a sentinel on the wire (score `-1.0`, anomaly `0`) -- "
                 "this folder can never be silently confused for the one scoring the other "
                 "head off the same backbone.")
    lines.append("")

    lines.append("## Active threshold (baked into `ssm_head_ref.c`)")
    lines.append("")
    chosen_thr, chosen_info = metrics_by_threshold[chosen_method]["threshold"], \
        metrics_by_threshold[chosen_method]["info"]
    chosen_m = metrics_by_threshold[chosen_method]["metrics"]
    const_name = (f"ssm_threshold_{'euclidean' if head == 'euclidean' else 'knn16'}"
                  + ("_sumsq" if is_int8_head else ""))
    lines.append(f"Default method: **`{chosen_method}`**, baked in as `{const_name}`"
                 f" = **{chosen_thr:.6f}**"
                 + (f" (float form). For the int8 head the decision uses the int32 "
                    f"`{const_name}` constant" if is_int8_head else "") + ".")
    lines.append("")
    lines.append(f"At this threshold on the case {held_out_case} test split: "
                 f"P={chosen_m['precision']:.3f}, R={chosen_m['recall']:.3f}, "
                 f"A={chosen_m['accuracy']:.3f}, F1={chosen_m['f1']:.3f}.")
    lines.append("")
    lines.append(f"Ranking quality (threshold-independent): AUC={auc:.4f}, pAUC={pauc:.4f}.")
    lines.append("")

    lines.append("## Swapping the threshold by hand")
    lines.append("")
    if is_int8_head:
        lines.append(f"Edit the `{const_name}` constant in `ssm_head_ref.c`. The int8 "
                     f"head decides on the int32 sum-of-squares form; the float score is "
                     f"reported but not used for the decision. To convert a float threshold "
                     f"`t` (from the table below) into the int32 form: "
                     f"`round((t / {final_norm_scale:.9g}) ** 2)`. Every method's int32 form "
                     f"is precomputed in the table.")
    else:
        lines.append(f"Edit the `{const_name}` constant in `ssm_head_ref.c` to any value "
                     f"from the table below, then rebuild. No re-export needed.")
    lines.append("")

    lines.append("## All 14 threshold methods")
    lines.append("")
    lines.append("Every method below was fit on this setup's own `calib_normal` scores "
                 "(same held-out machine, disjoint from train and from the balanced test "
                 "split), then evaluated on the test split. Numbers match "
                 "`diagnostics.json` in this folder exactly. The active default is marked.")
    lines.append("")
    if is_int8_head:
        lines.append("| Method | Threshold (float) | Threshold (int32 sumsq) | P | R | A | F1 | |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    else:
        lines.append("| Method | Threshold | P | R | A | F1 | |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | :--- |")
    for method_name, entry in metrics_by_threshold.items():
        m = entry["metrics"]
        thr = entry["threshold"]
        marker = "**<- active**" if method_name == chosen_method else ""
        if is_int8_head:
            sumsq = int(round((thr / final_norm_scale) ** 2)) if thr > 0 else -1
            lines.append(f"| `{method_name}` | {thr:.6f} | {sumsq} | {m['precision']:.3f} | "
                         f"{m['recall']:.3f} | {m['accuracy']:.3f} | {m['f1']:.3f} | {marker} |")
        else:
            lines.append(f"| `{method_name}` | {thr:.6f} | {m['precision']:.3f} | "
                         f"{m['recall']:.3f} | {m['accuracy']:.3f} | {m['f1']:.3f} | {marker} |")
    lines.append("")
    lines.append("Some methods (notably `evt_*` at aggressive false-alarm targets) can "
                 "collapse on certain score distributions -- see `findings/543`. A high "
                 "threshold with near-zero recall in the table above is that collapse, not "
                 "a backbone problem; the AUC is unaffected.")
    lines.append("")
    return "\n".join(lines)


# =====================================================================
# ORCHESTRATION
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--deploy-root", default="mcu/deploy")
    parser.add_argument("--only", default=None,
                        help="comma-separated setup identifiers to build (default: all 18)")
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

    fold = load_resolved_fold(base_dir)  # was: fold = get_fold(args.held_out_case)
    norm_mean, norm_std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    norm_stats = (norm_mean, norm_std)
    test_labels = (fold["test"]["label"].values == "anomaly").astype(int)

    setups = build_setup_matrix()
    if args.only:
        wanted = set(args.only.split(","))
        setups = [s for s in setups if s["identifier"] in wanted]
        if not setups:
            raise ValueError(f"--only matched no setups; valid ids: "
                             f"{[s['identifier'] for s in build_setup_matrix()]}")

    # Preload true-int8 shared calibration once (ranges + hooks are identical
    # across both h-widths and every true-int8 setup).
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

    # Embedding cache, keyed by backbone_key -- computed at most once each.
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

    # Head fit cache, keyed by (backbone_key, head) -- KMeans is expensive.
    head_cache = {}

    def get_head(setup, train_emb):
        key = (setup["backbone_key"], setup["head"])
        if key not in head_cache:
            head_cache[key] = fit_head(setup["head"], train_emb, seed)
        return head_cache[key]

    # State dict cache for weight emission, keyed by backbone_key.
    sd_cache = {}

    def get_state_dict(setup):
        key = setup["backbone_key"]
        if key in sd_cache:
            return sd_cache[key]
        model = SSMBackbone(**m)
        ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        # Always the ORIGINAL fp32 state dict -- both emission paths quantize
        # internally (fake-quant via fq_emit_*; true-int8 via
        # prepare_quantized_model), matching the archived exporters.
        sd = model.state_dict()
        sd_cache[key] = sd
        return sd

    deploy_root = PROJECT_ROOT / args.deploy_root
    setups_root = deploy_root / f"case{args.held_out_case}" / args.model_hash / "setups"
    setups_root.mkdir(parents=True, exist_ok=True)

    all_identifiers = [s["identifier"] for s in build_setup_matrix()]

    for setup in setups:
        combo_number = all_identifiers.index(setup["identifier"]) + 1
        ident = setup["identifier"]
        print(f"\n[{combo_number:2d}/18] {ident}")
        out_dir = setups_root / ident
        out_dir.mkdir(parents=True, exist_ok=True)

        emb = get_embeddings(setup)
        train_emb, test_emb, calib_emb = emb["train"], emb["test"], emb["calib_normal"]
        sd = get_state_dict(setup)

        # Emit weights + copy backbone.
        final_norm_scale = None
        if setup["family"] == "fake_quant":
            emit_fake_quant_weights(
                out_dir, cfg, sd, dims, norm_stats,
                setup["weight_mode"], setup["granularity"], setup["activation_group"],
                base_dir, args.held_out_case)
            if setup["activation_group"] == "boundaries":
                final_norm_scale = load_ranges_scales(base_dir)["final_norm_output"]
        else:
            final_norm_scale = emit_true_int8_weights(
                out_dir, cfg, sd, dims, norm_stats, ti_ranges, ti_hook_scales, setup["h_width"])

        # Fit head, score test + calib, compute AUC/pAUC + all 14 thresholds.
        ref, scorer = get_head(setup, train_emb)
        test_scores = scorer(test_emb)
        calib_scores = scorer(calib_emb)
        auc = float(roc_auc_score(test_labels, test_scores))
        pauc = float(roc_auc_score(test_labels, test_scores, max_fpr=0.1))

        thresholds = compute_all_thresholds(calib_scores, seed)
        metrics_by_threshold = {}
        for method_name, (thr, info) in thresholds.items():
            m_metrics = secondary_metrics(test_scores, test_labels, thr)
            metrics_by_threshold[method_name] = {
                "threshold": float(thr),
                "info": info,
                "metrics": {k: m_metrics[k] for k in ("precision", "recall", "accuracy", "f1")},
            }

        chosen_method = DEFAULT_THRESHOLD_METHOD
        if chosen_method not in metrics_by_threshold:
            raise KeyError(f"default method {chosen_method} not among computed thresholds")
        chosen_thr = metrics_by_threshold[chosen_method]["threshold"]

        # Emit head reference + module (single head, baked-in default threshold).
        emit_head_ref_and_module(out_dir, setup["head"], setup["head_precision"],
                                 ref, chosen_thr, final_norm_scale)

        # diagnostics.json -- machine-readable, all 14 methods, both threshold
        # forms for int8 heads so either can be swapped by hand.
        diagnostics = {
            "identifier": ident,
            "combo_number": combo_number,
            "config": args.config,
            "held_out_case": args.held_out_case,
            "model_hash": args.model_hash,
            "backbone": {k: setup[k] for k in setup
                        if k in ("family", "weight_mode", "granularity",
                                 "activation_group", "h_width")},
            "head": setup["head"],
            "head_precision": setup["head_precision"],
            "default_threshold_method": chosen_method,
            "final_norm_scale": (float(final_norm_scale) if final_norm_scale is not None else None),
            "auc": auc,
            "pauc": pauc,
            "thresholds": {},
        }
        for method_name, entry in metrics_by_threshold.items():
            thr = entry["threshold"]
            record = {
                "threshold": thr,
                "info": entry["info"],
                "metrics": entry["metrics"],
            }
            if setup["head_precision"] == "int8" and final_norm_scale:
                record["threshold_sumsq"] = (int(round((thr / final_norm_scale) ** 2))
                                            if thr > 0 else -1)
            diagnostics["thresholds"][method_name] = record

        with open(out_dir / "diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=4, default=str)

        readme = render_readme(setup, combo_number, args.config, args.held_out_case,
                               args.model_hash, metrics_by_threshold, chosen_method,
                               setup["head"], setup["head_precision"], auc, pauc,
                               final_norm_scale if final_norm_scale else 1.0)
        (out_dir / "README.md").write_text(readme)

        print(f"        AUC={auc:.4f} pAUC={pauc:.4f}  default '{chosen_method}' "
              f"thr={chosen_thr:.4f}  "
              f"F1={metrics_by_threshold[chosen_method]['metrics']['f1']:.3f}")

    print(f"\nDone. {len(setups)} setup folder(s) under {setups_root}")


if __name__ == "__main__":
    main()