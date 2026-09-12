"""
Deployment-matrix exporter for CLASSIC (selective=False) configs.

The selective counterpart is mcu/export_deploy_matrix.py. That file is NOT
modified by this one and does not import from it -- the dependency runs one
way only, so the selective path is provably untouched. Everything that is
backbone-agnostic (head fitting, the 14 threshold methods, the README
renderer, the C emission helpers, the single-head sentinel discipline) is
imported from it rather than reimplemented.

    mcu/deploy/case<N>/<model_hash>/setups/<setup_identifier>/

Same eight C files, README.md and diagnostics.json per folder, same layout,
same guarantees. The model hash differs from any selective run, so classic
and selective folders can never collide.

THE MATRIX -- 28 combos:

    8   weight-only int8, ALL weights
        granularity {per-channel, per-tensor} x recurrence x head
    4   weight int8 (all, per-tensor) + activation-boundaries int8
        recurrence x head
    16  true int8 arithmetic
        h-width {int8, int16} x recurrence x head x head-precision {fp32, int8}

`projections` is deliberately absent as a weight mode. On this branch it
would mean in_proj + conv + out_proj + norms only, which is a different set
from the selective matrix's `projections` and therefore not comparable
against it; with A/B/C/dt left fp32 it also is not a scheme anyone would
deploy. All eight weight-only folders quantize everything.

THE RECURRENCE AXIS -- the classic-only axis, replacing that slot:

    qab    A_log, dt and B ship as int8. The firmware dequantizes them and
           computes delta = softplus(dt), A = -exp(A_log),
           A_bar = 1 + clamp(delta*A, -1.9) and B_bar = delta*B every frame --
           the same arithmetic in the same order as the selective backbone,
           which is what makes the two matrices comparable.

    qabar  A_log, dt and B do NOT ship. A_bar and B_bar are discretized at
           export time and ship already quantized, so the firmware has no
           discretize step. Two d_inner x d_state arrays per layer instead
           of one, so more flash, but no per-frame expf/softplus/clamp.

Both forms are calibrated against the same ranges.json entries
(block<i>.A_bar / block<i>.B_bar), which is what keeps the h-width axis
comparable across them.

A NOTE ON HOW qabar's EMBEDDINGS ARE COMPUTED: there is no state-dict entry
for A_bar, so the fake-quant path cannot quantize it by perturbing weights.
It is quantized instead through ActivationQuantizer's `discretized_const`
group with scale_source='onthefly'. On this branch that is not an
approximation: A_bar and B_bar are constants, so the on-the-fly per-tensor
max IS the tensor's own max-abs, and ActivationQuantizer's per-channel axis
for '.A_bar'/'.B_bar' is already ndim-2, which is d_inner -- the same axis
fq_quantize_int8_perchannel reduces over. The two produce identical values.
Under qabar the state-dict pass skips A_log/dt/B so nothing is quantized
twice.

Usage:
    python mcu/export_deploy_matrix_classic.py \
        --config 352f70960ed3.yaml --held-out-case 1 --model-hash b77482e85dc3
    # optional: --only w_all_pertensor_qab_euclidean_fp32
    #           --deploy-root mcu/deploy
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.config import load_config_by_name, PROJECT_ROOT
from src.features.stats import compute_normalization_stats
from src.features.baselines import load_fold_clips, apply_normalization
from src.models.backbone import SSMBackbone
from src.eval.thresholds import secondary_metrics

from checks.smoke.quant.weight_quant_parity import should_quantize, quantize_dequantize
from checks.smoke.quant.activation_quant import ActivationQuantizer
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim_classic import (
    RECURRENCES, EULER_CLAMP, calibrate_missing_ranges_classic,
    prepare_quantized_model_classic, quantized_forward_classic,
)

from mcu.export_deploy_matrix import (
    load_resolved_fold, fit_head, compute_all_thresholds,
    emit_head_ref_and_module, render_readme,
    format_c_float, c_array, c_int8_array, ti_c_int8_array, ti_c_float_array,
    fq_quantize_int8_pertensor, fq_emit_2d_field, fq_emit_1d_field, fq_emit_A_field,
    FQ_BOUNDARY_FIELDS_PER_LAYER, DEFAULT_THRESHOLD_METHOD,
)

# Canonical, hand-maintained CLASSIC backbone sources -- copied verbatim,
# never generated. Separate from the selective pair: the recurrence really
# is different code, not a different header.
BACKBONE_SRC_CLASSIC_FAKE_QUANT = Path(__file__).parent / "ssm_backbone_classic_src"
BACKBONE_SRC_CLASSIC_TRUE_INT8 = Path(__file__).parent / "ssm_true_int8_classic_src"

# State-dict suffixes that do NOT ship under qabar, because the firmware
# never needs them once A_bar/B_bar are baked. C still ships (it is the scan
# readout, not part of the discretize step).
QABAR_EXCLUDED_SUFFIXES = (".A_log", ".dt", ".B")

TOTAL_COMBOS = 28


# =====================================================================
# SETUP MATRIX
# =====================================================================

def build_classic_setup_matrix():
    """Returns the 28 setups. backbone_key groups setups that share an
    identical backbone forward pass, so embeddings compute once per key:
    6 fake-quant passes (fast) and 4 true-int8 passes (slow)."""
    setups = []

    # Combos 1-8: weight-only int8, all weights.
    for granularity, g_tag in (("per-channel", "perchannel"), ("per-tensor", "pertensor")):
        for recurrence in RECURRENCES:
            for head in ("euclidean", "knn16"):
                setups.append({
                    "family": "fake_quant",
                    "weight_mode": "all",
                    "granularity": granularity,
                    "activation_group": "none",
                    "recurrence": recurrence,
                    "head": head,
                    "head_precision": "fp32",
                    "backbone_key": f"fake_all_{g_tag}_{recurrence}_none",
                    "identifier": f"w_all_{g_tag}_{recurrence}_{head}_fp32",
                })

    # Combos 9-12: weight all/per-tensor + activation boundaries.
    for recurrence in RECURRENCES:
        for head in ("euclidean", "knn16"):
            setups.append({
                "family": "fake_quant",
                "weight_mode": "all",
                "granularity": "per-tensor",
                "activation_group": "boundaries",
                "recurrence": recurrence,
                "head": head,
                "head_precision": "fp32",
                "backbone_key": f"fake_all_pertensor_{recurrence}_boundaries",
                "identifier": f"w_all_pertensor_actboundaries_{recurrence}_{head}_fp32",
            })

    # Combos 13-28: true int8.
    for h_width, h_tag in (("int8", "h8"), ("int16", "h16")):
        for recurrence in RECURRENCES:
            for head in ("euclidean", "knn16"):
                for head_precision in ("fp32", "int8"):
                    setups.append({
                        "family": "true_int8",
                        "h_width": h_width,
                        "recurrence": recurrence,
                        "head": head,
                        "head_precision": head_precision,
                        "backbone_key": f"true_int8_{h_tag}_{recurrence}",
                        "identifier": f"true_int8_{h_tag}_{recurrence}_{head}_{head_precision}",
                    })

    return setups


def classic_should_quantize(param_name, recurrence):
    """Every weight in the classic matrix is quantized (weight_mode='all'),
    minus whatever this recurrence form does not ship.

    Relies on WEIGHT_MODE_SUFFIXES['all'] having been extended with '.B',
    '.C' and '.dt' -- see the note in weight_quant_parity.py. Without that
    extension this silently leaves the whole static recurrence in fp32."""
    if not should_quantize(param_name, "all"):
        return False
    if recurrence == "qabar" and param_name.endswith(QABAR_EXCLUDED_SUFFIXES):
        return False
    return True


def classic_discretized_constants(sd, i):
    """A_bar and B_bar for one layer, in fp32, computed exactly as
    ssm_block.py's discretize(discretization='euler') computes them on the
    non-selective branch: delta = softplus(dt) and B are constants, so both
    outputs are (d_inner, d_state) with no batch or time axis."""
    A = -torch.exp(sd[f"blocks.{i}.A_log"])
    delta = F.softplus(sd[f"blocks.{i}.dt"])
    deltaA = delta.unsqueeze(-1) * A
    A_bar = 1.0 + torch.clamp(deltaA, min=EULER_CLAMP)
    B_bar = delta.unsqueeze(-1) * sd[f"blocks.{i}.B"].unsqueeze(-2)
    return A_bar, B_bar


class ChainedQuantizer:
    """Applies several ActivationQuantizers in sequence. Needed only for the
    qabar + activation-boundaries combos, where 'discretized_const' runs on
    on-the-fly scales and 'boundaries' runs on static ranges scales -- two
    different scale sources, which one ActivationQuantizer cannot express.
    The two groups are disjoint by construction, so order does not matter."""

    def __init__(self, quantizers):
        self.quantizers = [q for q in quantizers if q is not None]

    @property
    def applied_names(self):
        names = set()
        for q in self.quantizers:
            names |= q.applied_names
        return names

    def apply(self, name, tensor):
        for q in self.quantizers:
            tensor = q.apply(name, tensor)
        return tensor


# =====================================================================
# FAKE-QUANT WEIGHT EMISSION (ssm_weights.h/.c)
# The positional-struct-init discipline from export_deploy_matrix.py's
# emit_fake_quant_weights applies here unchanged and is load-bearing: a
# field's struct-member declaration and its initializer value MUST be
# appended in the same loop iteration, or every field after it silently
# misaligns. Do not "tidy" these into separate blocks.
# =====================================================================

def emit_classic_fake_quant_weights(out_dir, sd, dims, norm_stats, granularity,
                                    recurrence, activation_group, checkpoint_dir):
    d_model, d_state, d_inner, d_conv, n_layers = dims
    norm_mean, norm_std = norm_stats

    def is_quantized(sd_key):
        return classic_should_quantize(sd_key, recurrence)

    act_quantized = activation_group == "boundaries"
    act_scales = load_ranges_scales(checkpoint_dir) if act_quantized else {}

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
        ("in_proj_w", "SSM_IN_PROJ_W", lambda i: sd[f"blocks.{i}.in_proj.weight"],
         lambda i: f"blocks.{i}.in_proj.weight", "SSM_D_MODEL"),
        ("conv_w", "SSM_CONV_W", lambda i: sd[f"blocks.{i}.conv.weight"],
         lambda i: f"blocks.{i}.conv.weight", "SSM_D_CONV"),
        ("out_proj_w", "SSM_OUT_PROJ_W", lambda i: sd[f"blocks.{i}.out_proj.weight"],
         lambda i: f"blocks.{i}.out_proj.weight", "SSM_D_INNER"),
    ]
    fields_1d = [
        ("conv_b", "SSM_CONV_B", lambda i: sd[f"blocks.{i}.conv.bias"],
         lambda i: f"blocks.{i}.conv.bias"),
        ("D", "SSM_D_PARAM", lambda i: sd[f"blocks.{i}.D"],
         lambda i: f"blocks.{i}.D"),
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

        if recurrence == "qab":
            init_parts.append(fq_emit_A_field(
                f"blocks_{i}_A", "SSM_A", sd[f"blocks.{i}.A_log"],
                is_quantized(f"blocks.{i}.A_log"), granularity, decl_lines, *shape_lists))
            init_parts.append(fq_emit_1d_field(
                f"blocks_{i}_dt", "SSM_DT", sd[f"blocks.{i}.dt"],
                is_quantized(f"blocks.{i}.dt"), decl_lines, *shape_lists))
            init_parts.append(fq_emit_1d_field(
                f"blocks_{i}_B", "SSM_B", sd[f"blocks.{i}.B"],
                is_quantized(f"blocks.{i}.B"), decl_lines, *shape_lists))
        else:
            A_bar, B_bar = classic_discretized_constants(sd, i)
            # Always quantized: this matrix has no fp32-weight scheme, and
            # these two arrays are the entire point of the qabar variant.
            init_parts.append(fq_emit_2d_field(
                f"blocks_{i}_A_bar", "SSM_A_BAR_BAKED", A_bar, True,
                granularity, "SSM_D_STATE", decl_lines, *shape_lists))
            init_parts.append(fq_emit_2d_field(
                f"blocks_{i}_B_bar", "SSM_B_BAR_BAKED", B_bar, True,
                granularity, "SSM_D_STATE", decl_lines, *shape_lists))

        init_parts.append(fq_emit_1d_field(
            f"blocks_{i}_C", "SSM_C", sd[f"blocks.{i}.C"],
            is_quantized(f"blocks.{i}.C"), decl_lines, *shape_lists))

        if act_quantized:
            for name in FQ_BOUNDARY_FIELDS_PER_LAYER:
                shape_lists[0].append(f"    float {name}_scale;")
            for name, (key_fmt, _) in FQ_BOUNDARY_FIELDS_PER_LAYER.items():
                init_parts.append(format_c_float(act_scales[key_fmt.format(i=i)]))

        init_parts.append(emit_norm_instance(f"blocks_{i}_norm_w", sd[f"norms.{i}.weight"]))
        decl_lines.append("")
        block_inits.append(f"    {{ {', '.join(init_parts)} }}")

    # The recurrence form, expressed entirely as macros so that
    # ssm_backbone.c's text is identical for qab and qabar. delta_c is
    # hoisted per channel by the caller; the baked form discards it via the
    # comma operator, which also keeps -Wunused-variable quiet.
    if recurrence == "qab":
        macro_lines.append("#define SSM_DELTA(w, c) ssm_softplus(SSM_DT((w), (c)))")
        macro_lines.append("#define SSM_A_BAR(w, c, n, d) ssm_euler_abar((d) * SSM_A((w), (c), (n)))")
        macro_lines.append("#define SSM_B_BAR(w, c, n, d) ((d) * SSM_B((w), (n)))")
    else:
        macro_lines.append("#define SSM_DELTA(w, c) (0.0f)")
        macro_lines.append("#define SSM_A_BAR(w, c, n, d) ((void)(d), SSM_A_BAR_BAKED((w), (c), (n)))")
        macro_lines.append("#define SSM_B_BAR(w, c, n, d) ((void)(d), SSM_B_BAR_BAKED((w), (c), (n)))")

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
        final_norm_scale_decl = (f"const float ssm_final_norm_scale = "
                                 f"{format_c_float(act_scales['final_norm_output'])};")
        final_norm_scale_extern = "extern const float ssm_final_norm_scale;"
        macro_lines.append("#define SSM_QUANT_FINAL_NORM(val) (ssm_quant_dequant((val), ssm_final_norm_scale))")
    else:
        macro_lines.append("#define SSM_QUANT_FINAL_NORM(val) (val)")

    header = f"""#pragma once

#include <stdint.h>
#include <math.h>

/* CLASSIC (selective=False) backbone. No SSM_DT_RANK: this branch has no
 * x_proj/dt_proj, so there is no low-rank delta path to size. */

#define SSM_D_MODEL {d_model}
#define SSM_D_STATE {d_state}
#define SSM_D_INNER {d_inner}
#define SSM_D_CONV {d_conv}
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
        shutil.copy(BACKBONE_SRC_CLASSIC_FAKE_QUANT / fname, out_dir / fname)


# =====================================================================
# TRUE-INT8 WEIGHT EMISSION (ssm_weights.h/.c)
# Designated initializers throughout, so member order is not load-bearing
# here the way it is in the fake-quant emitter above.
# =====================================================================

TI_SCALE_FIELD_MAP_CLASSIC = {
    "norm_out": "s_norm_out",
    "conv_out": "s_conv_out",
    "u_post_conv_silu": "s_u_post_conv_silu",
    "z_gate": "s_z_gate",
    "B": "s_B",
    "C": "s_C",
    "delta": "s_delta",
    "A_bar": "s_A_bar",
    "B_bar": "s_B_bar",
    "h_scale": "s_h",
    "y_scan": "s_y_scan",
    "y_gated": "s_y_gated",
    "block_output": "s_block_output",
}


def emit_classic_true_int8_weights(out_dir, sd, dims, norm_stats, ranges, hook_scales,
                                   h_width, recurrence):
    d_model, d_state, d_inner, d_conv, n_layers = dims
    norm_mean, norm_std = norm_stats

    weights, scales, luts_per_layer = prepare_quantized_model_classic(
        sd, ranges, hook_scales, d_inner, n_layers, h_width, recurrence)

    h_width_define = "#define SSM_H_WIDTH_INT16\n" if h_width == "int16" else ""

    if recurrence == "qab":
        rec_members = ("\tconst int8_t *delta_q;\n"
                       "\tconst int8_t *B_q;\n"
                       "\tconst float *A;\n")
        rec_macros = "\n".join([
            "#define SSM_TI_DELTA_REAL(w, c) ((float) (w)->delta_q[(c)] * (w)->s_delta)",
            "#define SSM_TI_A_BAR_Q(w, c, n, d) "
            "ssm_requantize(ssm_euler_abar((d) * (w)->A[(c) * SSM_D_STATE + (n)]), (w)->s_A_bar)",
            "#define SSM_TI_B_BAR_Q(w, c, n, d) "
            "ssm_requantize((float) ((int32_t) (w)->delta_q[(c)] * (int32_t) (w)->B_q[(n)]) "
            "* (w)->s_delta * (w)->s_B, (w)->s_B_bar)",
        ])
    else:
        rec_members = ("\tconst int8_t *A_bar_q;\n"
                       "\tconst int8_t *B_bar_q;\n")
        rec_macros = "\n".join([
            "#define SSM_TI_DELTA_REAL(w, c) (0.0f)",
            "#define SSM_TI_A_BAR_Q(w, c, n, d) ((void) (d), (w)->A_bar_q[(c) * SSM_D_STATE + (n)])",
            "#define SSM_TI_B_BAR_Q(w, c, n, d) ((void) (d), (w)->B_bar_q[(c) * SSM_D_STATE + (n)])",
        ])

    header = f"""#pragma once

#include <stdint.h>

/* CLASSIC (selective=False), true int8 arithmetic. No SSM_DT_RANK and no
 * softplus LUT: delta = softplus(dt) is a constant on this branch, folded
 * into delta_q (qab) or into A_bar/B_bar (qabar) at export time. */

{h_width_define}#define SSM_D_MODEL {d_model}
#define SSM_D_STATE {d_state}
#define SSM_D_INNER {d_inner}
#define SSM_D_CONV {d_conv}
#define SSM_N_LAYERS {n_layers}

/* One layer's weights, scales and LUTs. Shape depends on the recurrence
 * form; ssm_backbone.c never names these members directly, only through the
 * SSM_TI_* macros below, so its text is identical for qab and qabar. */
typedef struct {{
\tconst int8_t *in_proj_u_w_q;
\tconst int8_t *in_proj_z_w_q;
\tfloat in_proj_w_scale;

\tconst int8_t *conv_w_q;
\tfloat conv_w_scale;
\tconst float *conv_b;

\tconst int8_t *out_proj_w_q;
\tfloat out_proj_w_scale;

\tconst int8_t *C_q;
\tconst float *D;
\tconst float *norm_w;

{rec_members}
\tfloat s_norm_out;
\tfloat s_conv_out;
\tfloat s_u_post_conv_silu;
\tfloat s_z_gate;
\tfloat s_B;
\tfloat s_C;
\tfloat s_delta;
\tfloat s_A_bar;
\tfloat s_B_bar;
\tfloat s_h;
\tfloat s_y_scan;
\tfloat s_y_gated;
\tfloat s_block_output;

\tconst int8_t *lut_silu_conv;
\tconst int8_t *lut_silu_z;
\tfloat s_silu_z;
}} ssm_true_int8_layer_t;

/* s_delta and s_B are emitted under both forms so this struct and
 * TI_SCALE_FIELD_MAP_CLASSIC stay uniform; under qabar nothing reads them. */

{rec_macros}

/* Deliberately does NOT #include "ssm_backbone.h" -- that header includes
 * THIS file for the macros, h-width flag, and this struct type. */

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
        decl_lines.append(ti_c_int8_array(f"{p}out_proj_w_q", w["out_proj_w_q"]))
        decl_lines.append(ti_c_int8_array(f"{p}C_q", w["C_q"]))
        decl_lines.append(ti_c_float_array(f"{p}D", w["D_real"]))
        decl_lines.append(ti_c_float_array(f"{p}norm_w", sd[f"norms.{i}.weight"].numpy()))

        if recurrence == "qab":
            decl_lines.append(ti_c_int8_array(f"{p}delta_q", w["delta_q"]))
            decl_lines.append(ti_c_int8_array(f"{p}B_q", w["B_q"]))
            decl_lines.append(ti_c_float_array(f"{p}A", w["A_real"]))
            rec_inits = f".delta_q = {p}delta_q, .B_q = {p}B_q, .A = {p}A,"
        else:
            decl_lines.append(ti_c_int8_array(f"{p}A_bar_q", w["A_bar_q"]))
            decl_lines.append(ti_c_int8_array(f"{p}B_bar_q", w["B_bar_q"]))
            rec_inits = f".A_bar_q = {p}A_bar_q, .B_bar_q = {p}B_bar_q,"

        decl_lines.append(ti_c_int8_array(f"{p}lut_silu_conv", lut["silu_conv"]))
        decl_lines.append(ti_c_int8_array(f"{p}lut_silu_z", lut["silu_z"]))
        decl_lines.append("")

        scale_inits = ", ".join(
            f".{field} = {format_c_float(s[key])}"
            for key, field in TI_SCALE_FIELD_MAP_CLASSIC.items())
        layer_inits.append(f"""    {{
        .in_proj_u_w_q = {p}in_proj_u_w_q, .in_proj_z_w_q = {p}in_proj_z_w_q,
        .in_proj_w_scale = {format_c_float(w["in_proj_w_scale"])},
        .conv_w_q = {p}conv_w_q, .conv_w_scale = {format_c_float(w["conv_w_scale"])},
        .conv_b = {p}conv_b,
        .out_proj_w_q = {p}out_proj_w_q, .out_proj_w_scale = {format_c_float(w["out_proj_w_scale"])},
        .C_q = {p}C_q, .D = {p}D, .norm_w = {p}norm_w,
        {rec_inits}
        {scale_inits},
        .lut_silu_conv = {p}lut_silu_conv, .lut_silu_z = {p}lut_silu_z,
        .s_silu_z = {format_c_float(lut["silu_z_scale"])},
    }}""")

    lines = ['#include "ssm_weights.h"', ""]
    lines.extend(decl_lines)
    lines.append("const ssm_true_int8_layer_t ssm_layers[SSM_N_LAYERS] = {")
    lines.append(",\n".join(layer_inits))
    lines.append("};")
    lines.append("")
    lines.append(ti_c_float_array("ssm_final_norm_w", sd["final_norm.weight"].numpy(),
                                  linkage="const float"))
    lines.append(f'const float ssm_final_norm_scale = {format_c_float(ranges["final_norm_output"])};')
    lines.append("")
    lines.append(ti_c_float_array("ssm_norm_mean", norm_mean, linkage="const float"))
    lines.append(ti_c_float_array("ssm_norm_std", norm_std, linkage="const float"))
    (out_dir / "ssm_weights.c").write_text("\n".join(lines) + "\n")

    for fname in ("ssm_backbone.h", "ssm_backbone.c"):
        shutil.copy(BACKBONE_SRC_CLASSIC_TRUE_INT8 / fname, out_dir / fname)

    return ranges["final_norm_output"]


# =====================================================================
# EMBEDDINGS -- one pass per distinct backbone_key.
# =====================================================================

def build_classic_fake_quant_embeddings(cfg, base_dir, fold, norm_stats, granularity,
                                        recurrence, activation_group):
    device = "cpu"
    mean, std = norm_stats
    model = SSMBackbone(**cfg["model"]).to(device)
    ckpt = torch.load(base_dir / "ckpt.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sd = model.state_dict()
    for name, w in sd.items():
        if classic_should_quantize(name, recurrence):
            deq, _ = quantize_dequantize(w, granularity)
            sd[name] = deq
    model.load_state_dict(sd)
    model.pooling = "mean"

    quantizers = []
    if recurrence == "qabar":
        # Exact, not approximate, on this branch -- see the module docstring.
        quantizers.append(ActivationQuantizer(
            group="discretized_const", granularity=granularity,
            scale_source="onthefly"))
    if activation_group != "none":
        act_scales = load_ranges_scales(base_dir)
        quantizers.append(ActivationQuantizer(
            group=activation_group, granularity="per-tensor",
            scale_source="ranges", scales=act_scales))
    quantizer = ChainedQuantizer(quantizers) if quantizers else None

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


def build_classic_true_int8_embeddings(cfg, base_dir, fold, norm_stats, dims, ranges,
                                       hook_scales, h_width, recurrence):
    d_model, d_state, d_inner, d_conv, n_layers = dims
    mean, std = norm_stats
    model = SSMBackbone(**cfg["model"])
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    weights, scales, luts_per_layer = prepare_quantized_model_classic(
        sd, ranges, hook_scales, d_inner, n_layers, h_width, recurrence)

    def embed(split):
        rows = fold[split]
        out = np.zeros((len(rows), d_model))
        for idx, (_, row) in enumerate(rows.iterrows()):
            log_mel = np.load(row["cache_path"]).astype(np.float32)
            x = apply_normalization(log_mel, mean, std).astype(np.float32)
            out[idx] = quantized_forward_classic(
                x, sd, weights, scales, luts_per_layer, ranges,
                d_model, d_inner, d_state, d_conv, n_layers, h_width)
            if (idx + 1) % 50 == 0:
                print(f"      {split}: {idx + 1}/{len(rows)}")
        return out

    print(f"    true_int8 classic [h={h_width}, {recurrence}] embeddings "
          f"(slow, per-frame simulator)...")
    return {"train": embed("train"), "test": embed("test"), "calib_normal": embed("calib_normal")}


# =====================================================================
# README ADDENDUM -- the recurrence axis, which render_readme cannot know
# about. Appended to the shared renderer's output rather than forking it.
# =====================================================================

RECURRENCE_DOC = {
    "qab": (
        "`A_log`, `dt` and `B` ship as int8. The firmware dequantizes them and "
        "computes `delta = softplus(dt)`, `A = -exp(A_log)`, "
        "`A_bar = 1 + clamp(delta*A, -1.9)` and `B_bar = delta*B` on every frame, "
        "in the same order as the selective backbone. Cheaper in flash, more work "
        "per frame."),
    "qabar": (
        "`A_log`, `dt` and `B` do not ship at all. `A_bar` and `B_bar` are "
        "discretized at export time and ship already quantized, so the firmware "
        "has no discretize step. Two `SSM_D_INNER x SSM_D_STATE` arrays per layer "
        "instead of one, so more flash, but no per-frame `expf`, `softplus` or "
        "clamp."),
}

SHIPPED_TENSORS = {
    "qab": "`in_proj`, `conv`, `out_proj`, `D`, `A_log`, `dt`, `B`, `C`, RMSNorm weights",
    "qabar": "`in_proj`, `conv`, `out_proj`, `D`, `A_bar`, `B_bar`, `C`, RMSNorm weights",
}


def classic_recurrence_section(setup):
    recurrence = setup["recurrence"]
    lines = [
        "## Recurrence form: `" + recurrence + "`",
        "",
        "This is the classic (`selective=False`) matrix. Its extra axis, which the "
        "selective matrix does not have, is how the discretized recurrence "
        "coefficients reach the device.",
        "",
        RECURRENCE_DOC[recurrence],
        "",
        f"Tensors in this folder's `ssm_weights.c`: {SHIPPED_TENSORS[recurrence]}.",
        "",
        "Both forms are calibrated against the same `ranges.json` entries "
        "(`block<i>.A_bar`, `block<i>.B_bar`), so AUC and the h-width axis stay "
        "comparable between them and against the selective matrix.",
        "",
        "`ssm_backbone.c` is byte-for-byte identical across both forms: the "
        "difference lives entirely in `ssm_weights.h`'s `SSM_DELTA` / `SSM_A_BAR` / "
        "`SSM_B_BAR` macros.",
        "",
    ]
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
                        help=f"comma-separated setup identifiers to build "
                             f"(default: all {TOTAL_COMBOS})")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    seed = cfg["seed"]

    if m["selective"]:
        raise ValueError("this matrix handles selective=False only; "
                         "use mcu/export_deploy_matrix.py for selective configs")
    if m["discretization"] != "euler":
        raise ValueError(f"classic backbones assume euler; config says {m['discretization']!r}")

    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dims = (d_model, d_state, d_inner, d_conv, n_layers)

    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    if not (base_dir / "ckpt.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {base_dir/'ckpt.pt'}")

    fold = load_resolved_fold(base_dir)
    norm_mean, norm_std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    norm_stats = (norm_mean, norm_std)
    test_labels = (fold["test"]["label"].values == "anomaly").astype(int)

    setups = build_classic_setup_matrix()
    if args.only:
        wanted = set(args.only.split(","))
        setups = [s for s in setups if s["identifier"] in wanted]
        if not setups:
            raise ValueError(f"--only matched no setups; valid ids: "
                             f"{[s['identifier'] for s in build_classic_setup_matrix()]}")

    # True-int8 shared calibration, once. Independent of h-width and of
    # recurrence form (both read the same ranges.json entries).
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

    sd_cache = {}

    def get_state_dict():
        if "sd" not in sd_cache:
            model = SSMBackbone(**m)
            ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.eval()
            # Always the ORIGINAL fp32 state dict -- both emission paths
            # quantize internally.
            sd_cache["sd"] = model.state_dict()
        return sd_cache["sd"]

    deploy_root = PROJECT_ROOT / args.deploy_root
    setups_root = deploy_root / f"case{args.held_out_case}" / args.model_hash / "setups"
    setups_root.mkdir(parents=True, exist_ok=True)

    all_identifiers = [s["identifier"] for s in build_classic_setup_matrix()]

    for setup in setups:
        combo_number = all_identifiers.index(setup["identifier"]) + 1
        ident = setup["identifier"]
        print(f"\n[{combo_number:2d}/{TOTAL_COMBOS}] {ident}")
        out_dir = setups_root / ident
        out_dir.mkdir(parents=True, exist_ok=True)

        emb = get_embeddings(setup)
        train_emb, test_emb, calib_emb = emb["train"], emb["test"], emb["calib_normal"]
        sd = get_state_dict()

        final_norm_scale = None
        if setup["family"] == "fake_quant":
            emit_classic_fake_quant_weights(
                out_dir, sd, dims, norm_stats, setup["granularity"],
                setup["recurrence"], setup["activation_group"], base_dir)
            if setup["activation_group"] == "boundaries":
                final_norm_scale = load_ranges_scales(base_dir)["final_norm_output"]
        else:
            final_norm_scale = emit_classic_true_int8_weights(
                out_dir, sd, dims, norm_stats, ti_ranges, ti_hook_scales,
                setup["h_width"], setup["recurrence"])

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

        emit_head_ref_and_module(out_dir, setup["head"], setup["head_precision"],
                                 ref, chosen_thr, final_norm_scale)

        diagnostics = {
            "identifier": ident,
            "combo_number": combo_number,
            "total_combos": TOTAL_COMBOS,
            "matrix": "classic",
            "config": args.config,
            "held_out_case": args.held_out_case,
            "model_hash": args.model_hash,
            "selective": False,
            "backbone": {k: setup[k] for k in setup
                         if k in ("family", "weight_mode", "granularity",
                                  "activation_group", "h_width", "recurrence")},
            "quantized_state_dict_keys": sorted(
                k for k in sd if classic_should_quantize(k, setup["recurrence"])),
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
        readme = readme + classic_recurrence_section(setup)
        (out_dir / "README.md").write_text(readme)

        print(f"        AUC={auc:.4f} pAUC={pauc:.4f}  default '{chosen_method}' "
              f"thr={chosen_thr:.4f}  "
              f"F1={metrics_by_threshold[chosen_method]['metrics']['f1']:.3f}")

    print(f"\nDone. {len(setups)} setup folder(s) under {setups_root}")


if __name__ == "__main__":
    main()