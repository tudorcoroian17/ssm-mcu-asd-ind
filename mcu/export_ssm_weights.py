"""
Export a trained SSMBackbone checkpoint to plain C weight arrays for the
Nucleo-H7S3L8 Phase 4/5 port (05_phase_4_backbone_port.md step 3;
06_phase_5_quantization_and_head.md's deployment matrix).

GENERATED OUTPUT: ssm_weights.h and ssm_weights.c. Don't hand-edit either;
rerun this script if the checkpoint, config, fold, or quantization flags
change.

Also exports this fold's per-channel normalization stats (mean/std) --
the model runs on apply_normalization(log_mel, mean, std)
(src/features/baselines.py), not on raw log-mel. Computed the same way
src/eval/parity_vectors.py computes it, so these arrays match that fold's
parity_vectors*.npz "norm_mean"/"norm_std" exactly, not just approximately.

--dtype fp32 (the default) stores every weight as a plain float array, same
values as always. --dtype int8 quantizes the tensors --weight-mode selects
(mirroring checks/smoke/quant/weight_quant_parity.py's WEIGHT_MODE_SUFFIXES
exactly -- imported directly, so the two cannot disagree about which tensors
a mode covers) at --granularity.

DESIGN -- per-access dequantization, no boot-time step, no scratch RAM:
every weight stays exactly where it's exported (a plain float array, or an
int8 array + scale, always in flash/.rodata). ssm_backbone.c never reads a
struct field directly -- it calls an SSM_<FIELD>(w, ...) macro, and this
script defines that macro two different ways depending on whether the field
was quantized in THIS export:
  not quantized:  #define SSM_X(w, ...) ((w)->x[...])
  quantized:      #define SSM_X(w, ...) ((float)(w)->x_q[...] * (w)->x_scale[...])
ssm_backbone.c's source text is identical for every scheme; only these
macro definitions, generated here, differ. There is no runtime "is this
quantized" branch anywhere -- the choice is fixed at compile time by which
macro got emitted -- and no lazy one-time initialization step, since every
pointer is a compile-time constant.

Quantization applies uniformly to a field across every layer (never "block0
quantized, block1 not") since WEIGHT_MODE_SUFFIXES matches by field-type
name, not layer index. This is what lets the struct's shape and every
macro's definition be decided once, from layer 0, and reused for every
other layer -- the per-layer loop below only differs in which layer's real
data gets declared, never in the shape of what it declares.

Per-channel granularity scales along axis 0 of the tensor, which is this
field's row axis in every weight matrix here -- the same loop variable
(`o` or `c`) already used at each access site in ssm_backbone.c, so the
macro's row argument doubles as the per-channel scale index with no extra
plumbing. 1D fields (biases, D, norm weights) are always per-tensor even
under --granularity per-channel, matching
checks/smoke/quant/weight_quant_parity.py's fallback for ndim < 2.

Usage:
    python mcu/export_ssm_weights.py \
        --checkpoint runs/case1/16662b29beb3/ckpt.pt \
        --config f4cd557b7e3b.yaml \
        --held-out-case 1 \
        --out-dir mcu/deploy/case1/weight_int8_proj_pertensor \
        --dtype int8 --weight-mode projections --granularity per-tensor
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config_by_name
from src.data.folds import get_fold
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone

from checks.smoke.quant.weight_quant_parity import (
    should_quantize, WEIGHT_MODES, GRANULARITIES,
)


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


def quantize_int8_pertensor(w_np):
    """Symmetric int8, one scale for the whole tensor. Matches
    checks/smoke/quant/weight_quant_parity.py's quantize_dequantize math
    exactly -- keep in sync if either changes. Scale is always returned as a
    length-1 array, even here, so generated macros index _scale[0]
    uniformly regardless of granularity."""
    max_abs = float(np.max(np.abs(w_np)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1e-12
    q = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return q, np.array([scale], dtype=np.float32)


def quantize_int8_perchannel(w_np):
    """Symmetric int8, one scale per axis-0 row. Matches quantize_dequantize's
    per-channel granularity. Caller must not call this for tensors with
    fewer than two dimensions."""
    reduce_axes = tuple(range(1, w_np.ndim))
    max_abs = np.max(np.abs(w_np), axis=reduce_axes)
    scale = np.where(max_abs == 0.0, 1e-12, max_abs / 127.0).astype(np.float32)
    scale_bc = scale.reshape((-1,) + (1,) * (w_np.ndim - 1))
    q = np.clip(np.round(w_np / scale_bc), -127, 127).astype(np.int8)
    return q, scale


def emit_2d_field(field_name, macro_name, tensor, quantize, granularity, ncols_macro,
                  decl_lines, member_decls, macro_lines):
    """Row-major 2D field: SSM_<NAME>(w, r, c). r is also the per-channel
    scale index when quantized per-channel -- see module docstring.
    field_name must already be layer-prefixed (e.g. 'blocks_0_in_proj_w') --
    declares this layer's real data. member_decls/macro_lines should only be
    the real (shared) lists for one layer (layer 0); pass fresh throwaway
    lists for every other layer so the struct/macro text isn't duplicated.
    Returns the init-list expression for this field."""
    flat = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    bare = field_name.split('_', 2)[-1]

    if not quantize:
        decl_lines.append(c_array(field_name, flat))
        member_decls.append(f"    const float *{bare};")
        macro_lines.append(
            f"#define {macro_name}(w, r, c) ((w)->{bare}[(r) * {ncols_macro} + (c)])"
        )
        return field_name

    use_perchannel = granularity == "per-channel" and flat.ndim >= 2
    q, scale = (quantize_int8_perchannel(flat) if use_perchannel
               else quantize_int8_pertensor(flat))

    decl_lines.append(c_int8_array(f"{field_name}_q", q))
    decl_lines.append(c_array(f"{field_name}_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_q;")
    member_decls.append(f"    const float *{bare}_scale;")

    scale_idx = "(r)" if use_perchannel else "0"
    macro_lines.append(
        f"#define {macro_name}(w, r, c) "
        f"((float)(w)->{bare}_q[(r) * {ncols_macro} + (c)] * (w)->{bare}_scale[{scale_idx}])"
    )
    return f"{field_name}_q, {field_name}_scale"


def emit_1d_field(field_name, macro_name, tensor, quantize,
                  decl_lines, member_decls, macro_lines):
    """1D field: SSM_<NAME>(w, i). Always per-tensor, even under
    --granularity per-channel -- see module docstring. field_name must
    already be layer-prefixed. Returns the init-list expression."""
    flat = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    bare = field_name.split('_', 2)[-1]

    if not quantize:
        decl_lines.append(c_array(field_name, flat))
        member_decls.append(f"    const float *{bare};")
        macro_lines.append(f"#define {macro_name}(w, i) ((w)->{bare}[(i)])")
        return field_name

    q, scale = quantize_int8_pertensor(flat)
    decl_lines.append(c_int8_array(f"{field_name}_q", q))
    decl_lines.append(c_array(f"{field_name}_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_q;")
    member_decls.append(f"    const float *{bare}_scale;")
    macro_lines.append(
        f"#define {macro_name}(w, i) ((float)(w)->{bare}_q[(i)] * (w)->{bare}_scale[0])"
    )
    return f"{field_name}_q, {field_name}_scale"


def emit_A_field(field_name, macro_name, A_log_tensor, quantize, granularity,
                 decl_lines, member_decls, macro_lines):
    """A is never stored directly -- the checkpoint holds A_log. Not
    quantized: precompute A = -exp(A_log) once at export time, as before.
    Quantized: store A_log_q/A_log_scale (findings/521 -- A_log is the
    tensor that must be quantized, a different, smoother quantity than the
    exponentiated A) and apply -exp() inside the macro, every access.
    field_name must already be layer-prefixed (e.g. 'blocks_0_A'). Returns
    the init-list expression for this field."""
    flat = np.asarray(A_log_tensor.detach().cpu().numpy(), dtype=np.float32)
    bare = field_name.split('_', 2)[-1]

    if not quantize:
        A = -np.exp(flat)
        decl_lines.append(c_array(field_name, A))
        member_decls.append(f"    const float *{bare};")
        macro_lines.append(
            f"#define {macro_name}(w, r, c) ((w)->{bare}[(r) * SSM_D_STATE + (c)])"
        )
        return field_name

    use_perchannel = granularity == "per-channel" and flat.ndim >= 2
    q, scale = (quantize_int8_perchannel(flat) if use_perchannel
               else quantize_int8_pertensor(flat))

    decl_lines.append(c_int8_array(f"{field_name}_log_q", q))
    decl_lines.append(c_array(f"{field_name}_log_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_log_q;")
    member_decls.append(f"    const float *{bare}_log_scale;")

    scale_idx = "(r)" if use_perchannel else "0"
    macro_lines.append(
        f"#define {macro_name}(w, r, c) "
        f"(-expf((float)(w)->{bare}_log_q[(r) * SSM_D_STATE + (c)] * "
        f"(w)->{bare}_log_scale[{scale_idx}]))"
    )
    return f"{field_name}_log_q, {field_name}_log_scale"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--weight-mode", choices=WEIGHT_MODES, default="projections",
                        help="only meaningful when --dtype int8")
    parser.add_argument("--granularity", choices=GRANULARITIES, default="per-tensor",
                        help="only meaningful when --dtype int8")
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

    def is_quantized(sd_key):
        return args.dtype == "int8" and should_quantize(sd_key, args.weight_mode)

    # norms.{i}.weight and final_norm.weight all match WEIGHT_MODE_SUFFIXES'
    # 'norms.' pattern identically regardless of i, so this one check (from
    # block 0) is valid for every norm instance -- same uniformity argument
    # as the field loop below.
    norm_quantized = is_quantized("norms.0.weight")
    norm_member_decls = []
    if not norm_quantized:
        norm_member_decls.append("    const float *w;")
        norm_macro = "#define SSM_NORM_W(nw, i) ((nw)->w[(i)])"
    else:
        norm_member_decls.append("    const int8_t *w_q;")
        norm_member_decls.append("    const float *w_scale;")
        norm_macro = "#define SSM_NORM_W(nw, i) ((float)(nw)->w_q[(i)] * (nw)->w_scale[0])"

    def emit_norm_instance(field_name, tensor):
        flat = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        if not norm_quantized:
            decl_lines.append(c_array(field_name, flat))
            return f"{{ {field_name} }}"
        q, scale = quantize_int8_pertensor(flat)
        decl_lines.append(c_int8_array(f"{field_name}_q", q))
        decl_lines.append(c_array(f"{field_name}_scale", scale))
        return f"{{ {field_name}_q, {field_name}_scale }}"

    member_decls = []
    macro_lines = []
    decl_lines = []
    block_inits = []

    fields_2d = [
        ("in_proj_w", "SSM_IN_PROJ_W", lambda i: sd[f"blocks.{i}.in_proj.weight"],
         lambda i: f"blocks.{i}.in_proj.weight", "SSM_D_MODEL"),
        ("conv_w", "SSM_CONV_W", lambda i: sd[f"blocks.{i}.conv.weight"],
         lambda i: f"blocks.{i}.conv.weight", "SSM_D_CONV"),
        ("x_proj_w", "SSM_X_PROJ_W", lambda i: sd[f"blocks.{i}.x_proj.weight"],
         lambda i: f"blocks.{i}.x_proj.weight", "SSM_D_INNER"),
        ("dt_proj_w", "SSM_DT_PROJ_W", lambda i: sd[f"blocks.{i}.dt_proj.weight"],
         lambda i: f"blocks.{i}.dt_proj.weight", "SSM_DT_RANK"),
        ("out_proj_w", "SSM_OUT_PROJ_W", lambda i: sd[f"blocks.{i}.out_proj.weight"],
         lambda i: f"blocks.{i}.out_proj.weight", "SSM_D_INNER"),
    ]
    fields_1d = [
        ("conv_b", "SSM_CONV_B", lambda i: sd[f"blocks.{i}.conv.bias"],
         lambda i: f"blocks.{i}.conv.bias"),
        ("dt_proj_b", "SSM_DT_PROJ_B", lambda i: sd[f"blocks.{i}.dt_proj.bias"],
         lambda i: f"blocks.{i}.dt_proj.bias"),
        ("D", "SSM_D_PARAM", lambda i: sd[f"blocks.{i}.D"],
         lambda i: f"blocks.{i}.D"),
    ]

    # One loop, every layer -- declares each layer's real data, and (for
    # layer 0 only) also collects the struct member declarations and macro
    # definitions, which are identical for every layer by construction.
    for i in range(n_layers):
        init_parts = []
        shape_lists = (member_decls, macro_lines) if i == 0 else ([], [])

        for field_name, macro_name, tensor_fn, key_fn, ncols in fields_2d:
            init_parts.append(emit_2d_field(
                f"blocks_{i}_{field_name}", macro_name, tensor_fn(i), is_quantized(key_fn(i)),
                args.granularity, ncols, decl_lines, *shape_lists))
        for field_name, macro_name, tensor_fn, key_fn in fields_1d:
            init_parts.append(emit_1d_field(
                f"blocks_{i}_{field_name}", macro_name, tensor_fn(i), is_quantized(key_fn(i)),
                decl_lines, *shape_lists))
        init_parts.append(emit_A_field(
            f"blocks_{i}_A", "SSM_A", sd[f"blocks.{i}.A_log"], is_quantized(f"blocks.{i}.A_log"),
            args.granularity, decl_lines, *shape_lists))
        init_parts.append(emit_norm_instance(f"blocks_{i}_norm_w", sd[f"norms.{i}.weight"]))

        decl_lines.append("")
        block_inits.append(f"    {{ {', '.join(init_parts)} }}")

    final_norm_init = emit_norm_instance("ssm_final_norm_w_data", sd["final_norm.weight"])

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

/* Weight access -- every field, quantized or not, goes through one of these.
 * Which branch got emitted (plain float, or int8+scale) depends only on
 * this export's --dtype/--weight-mode/--granularity; ssm_backbone.c's
 * source text never changes across schemes. */
{chr(10).join(macro_lines)}
{norm_macro}

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const ssm_norm_weights_t ssm_final_norm_w;

/* Per-channel z-score stats for this fold's training data (held-out case
 * {args.held_out_case}). Apply as (raw - ssm_norm_mean) / ssm_norm_std
 * BEFORE feeding a log-mel frame into the backbone -- see
 * SSMBackbone_NormalizeFrame() in ssm_backbone.h. */
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
    lines.append("")
    lines.append(c_array("ssm_norm_mean", norm_mean, linkage="const float"))
    lines.append(c_array("ssm_norm_std", norm_std, linkage="const float"))

    (out_dir / "ssm_weights.c").write_text("\n".join(lines))

    print(f"Wrote {out_dir/'ssm_weights.h'} and {out_dir/'ssm_weights.c'}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    mode_str = f"{args.weight_mode}/{args.granularity}" if args.dtype == "int8" else "n/a"
    print(f"dtype={args.dtype}  weight_mode/granularity={mode_str}")


if __name__ == "__main__":
    main()