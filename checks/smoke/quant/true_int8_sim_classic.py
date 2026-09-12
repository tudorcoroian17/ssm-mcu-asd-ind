"""
True int8 quantized-inference simulator -- CLASSIC (selective=False) branch.

The selective counterpart is checks/smoke/quant/true_int8_sim.py. This file
does NOT reimplement it: every quantization primitive (quantize_int8,
quantize_weight_pertensor, quantized_linear, build_lut, apply_lut,
silu_fp32, rmsnorm_fp32) and H_WIDTHS are imported from it unchanged, so
the two branches cannot drift on the arithmetic they share. Only the parts
that genuinely differ live here.

WHAT DIFFERS FROM THE SELECTIVE BRANCH:

  1. No x_proj, no dt_proj. On this branch delta, B and C are static
     nn.Parameters (ssm_block.py's `else` arm), so there is nothing to
     project per frame and nothing to requantize out of a split.

  2. No softplus LUT. delta = softplus(dt) is a constant, computed once
     here in fp32 and quantized against s_delta. The selective branch needs
     a LUT because its delta_raw is data-dependent. Both SiLU LUTs
     (conv output, z gate) are still needed and are built identically.

  3. calibrate_missing_ranges_classic hooks only conv and the RMSNorms.
     The selective version also hooks x_proj/dt_proj to recover
     x_proj_delta_low_out and dt_proj_out; neither range exists on this
     branch and neither is referenced below.

  4. TWO RECURRENCE FORMS, selected by `recurrence`:

       qab    A_log, dt and B are quantized to int8 and SHIPPED. The device
              dequantizes them and computes delta = softplus(dt),
              A = -exp(A_log), A_bar = 1 + clamp(delta*A, -1.9),
              B_bar = delta*B every frame -- the same arithmetic, in the
              same order, as the selective backbone.

       qabar  A_log, dt and B are NOT shipped. A_bar and B_bar are
              discretized here in fp32 and shipped already quantized. The
              device does no discretize step at all.

     Both forms are calibrated against the SAME ranges.json entries
     (block<i>.A_bar / block<i>.B_bar), which is what keeps the h-width
     axis comparable across them and against the selective matrix.

PRECOMPUTED A_bar/B_bar UNDER qab -- WHY THIS IS NOT A SHORTCUT: on this
branch delta, A and B are constants, so A_bar and B_bar are constant too.
This file therefore computes their int8 values ONCE, in
classic_recurrence_constants(), for BOTH recurrence forms. Under qab the C
firmware really does recompute them per frame (see
ssm_true_int8_classic_src/ssm_backbone.c), but from the same inputs through
the same deterministic float ops, so the values are bit-identical and the
simulation stays faithful. The only thing hoisting changes is runtime: this
simulator is the slow path in the deployment matrix, and the recurrence is
its inner loop.

DELIBERATE EXCEPTIONS: identical to the selective branch -- RMSNorm
arithmetic stays fp32, log-mel is untouched, biases are added in real units
at each rescale point rather than separately quantized.

Loads existing checkpoints, does not train.

Usage:
    python -m checks.smoke.quant.true_int8_sim_classic \
        --config 352f70960ed3.yaml --held-out-case 1 --model-hash b77482e85dc3 \
        --recurrence qab --h-width int8
"""
import argparse

import numpy as np
import torch

from src.config import load_config_by_name, PROJECT_ROOT
from src.data.folds import get_fold
from src.features.baselines import apply_normalization
from src.features.stats import compute_normalization_stats
from src.models.backbone import SSMBackbone
from src.eval.parity_vectors import select_parity_clip
from checks.smoke.quant.activation_quant_parity import load_ranges_scales
from checks.smoke.quant.true_int8_sim import (
    H_WIDTHS, quantize_int8, quantize_weight_pertensor, quantized_linear,
    build_lut, apply_lut, silu_fp32, rmsnorm_fp32,
)

RECURRENCES = ("qab", "qabar")

# ssm_block.py's discretize() clamps delta*A at this floor before the euler
# step (1 + deltaA), keeping A_bar above -0.9 so the recurrence cannot
# diverge. Mirrored here and in both classic C backbones.
EULER_CLAMP = -1.9


def softplus_np(x):
    """Matches torch.nn.functional.softplus's default threshold=20: linear
    above 20, log1p(exp(x)) below. The np.minimum guard keeps exp() from
    overflowing on the branch that then discards its result."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


# ---------------------------------------------------------------------------
# Calibration -- ranges.json plus the two hook-derived ranges it lacks.
# ---------------------------------------------------------------------------

def calibrate_missing_ranges_classic(model, fold, mean, std, n_layers, n_clips=8):
    """conv's OUTPUT (the SiLU-at-conv LUT's input range) and each block's
    RMSNorm output (x_norm feeding that block). The selective branch's other
    two hooks (x_proj, dt_proj) have no counterpart here."""
    captured = {}
    for i in range(n_layers):
        for key in ("conv_out", "norm_out"):
            captured[f"block{i}.{key}"] = []

    hooks = []
    for i, block in enumerate(model.blocks):
        def conv_hook(module, inp, out, i=i):
            captured[f"block{i}.conv_out"].append(out.detach().abs().max().item())

        def norm_hook(module, inp, out, i=i):
            captured[f"block{i}.norm_out"].append(out.detach().abs().max().item())

        hooks.append(block.conv.register_forward_hook(conv_hook))
        hooks.append(model.norms[i].register_forward_hook(norm_hook))

    clips = fold["train"].head(n_clips)
    with torch.no_grad():
        for _, row in clips.iterrows():
            log_mel = np.load(row["cache_path"]).astype(np.float32)
            x = apply_normalization(log_mel, mean, std).astype(np.float32)
            model(torch.from_numpy(x).unsqueeze(0), mode="sequence", streaming=True)

    for h in hooks:
        h.remove()

    scales = {}
    for key, vals in captured.items():
        max_abs = max(vals) if vals else 1e-6
        scales[key] = max_abs / 127.0
    return scales


# ---------------------------------------------------------------------------
# Recurrence constants -- the whole qab/qabar difference, in one function.
# ---------------------------------------------------------------------------

def classic_recurrence_constants(sd, i, recurrence, s):
    """Returns every constant the recurrence needs, for one layer.

    Always returns A_bar_q, B_bar_q (int8, on s['A_bar'] / s['B_bar']) and
    C_q -- these are what the simulator's inner loop consumes, regardless of
    form. Under qab it ALSO returns delta_q, B_q and A_real, which are the
    tensors that actually ship to the device in that form; the exporter
    emits those and lets the firmware rebuild A_bar/B_bar from them.

    Scale note: delta, B and C are quantized against their ranges.json
    scales (s['delta'], s['B'], s['C']), not against a freshly derived
    per-tensor scale. For a STATIC parameter those are the same number --
    the recorder saw exactly the parameter's own values, so max|recorded| is
    max|parameter| -- so this is both faithful to the selective path (which
    must use the calibrated activation scale) and equivalent to per-tensor
    weight quantization. Verifiable directly against ranges.json.
    """
    if recurrence not in RECURRENCES:
        raise ValueError(f"unknown recurrence {recurrence!r}; expected one of {RECURRENCES}")

    A_log = sd[f"blocks.{i}.A_log"].numpy().astype(np.float32)
    dt = sd[f"blocks.{i}.dt"].numpy().astype(np.float32)
    B = sd[f"blocks.{i}.B"].numpy().astype(np.float32)
    C = sd[f"blocks.{i}.C"].numpy().astype(np.float32)

    out = {"C_q": np.clip(np.round(C / s["C"]), -127, 127).astype(np.int32)}

    if recurrence == "qab":
        # A_log is quantized per-tensor and then exponentiated, exactly as
        # the selective branch's load_quantized_weights does.
        A_log_q, A_log_scale = quantize_weight_pertensor(A_log)
        A_real = -np.exp(A_log_q * A_log_scale)
        delta_q = np.clip(np.round(softplus_np(dt) / s["delta"]), -127, 127).astype(np.int32)
        B_q = np.clip(np.round(B / s["B"]), -127, 127).astype(np.int32)

        delta_real = delta_q * s["delta"]
        deltaA_real = delta_real[:, None] * A_real
        A_bar_real = 1.0 + np.maximum(deltaA_real, EULER_CLAMP)
        # Raw int product first, scaled after -- matches the selective
        # branch's B_bar line and the firmware macro, not delta_real*B_real.
        B_bar_real = (delta_q[:, None].astype(np.int64) * B_q[None, :].astype(np.int64)) \
            .astype(np.float64) * s["delta"] * s["B"]

        out["A_real"] = A_real
        out["delta_q"] = delta_q
        out["B_q"] = B_q
    else:  # qabar -- nothing upstream of the discretize step ever ships
        A_real = -np.exp(A_log)
        delta_real = softplus_np(dt)
        deltaA_real = delta_real[:, None] * A_real
        A_bar_real = 1.0 + np.maximum(deltaA_real, EULER_CLAMP)
        B_bar_real = delta_real[:, None] * B[None, :]

    out["A_bar_q"] = np.clip(np.round(A_bar_real / s["A_bar"]), -127, 127).astype(np.int32)
    out["B_bar_q"] = np.clip(np.round(B_bar_real / s["B_bar"]), -127, 127).astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Weight loading -- everything quantized, per-tensor.
# ---------------------------------------------------------------------------

def load_quantized_weights_classic(sd, i, d_inner, recurrence, s):
    w = {}
    for name, key in [
        ("in_proj_w", f"blocks.{i}.in_proj.weight"),
        ("conv_w", f"blocks.{i}.conv.weight"),
        ("out_proj_w", f"blocks.{i}.out_proj.weight"),
        ("norm_w", f"norms.{i}.weight"),
    ]:
        q, scale = quantize_weight_pertensor(sd[key].numpy())
        w[f"{name}_q"], w[f"{name}_scale"] = q, scale

    w["conv_w_q"] = w["conv_w_q"].reshape(d_inner, -1)  # (d_inner,1,d_conv) -> (d_inner,d_conv)
    w["in_proj_u_q"] = w["in_proj_w_q"][:d_inner]
    w["in_proj_z_q"] = w["in_proj_w_q"][d_inner:]
    w["conv_b_fp32"] = sd[f"blocks.{i}.conv.bias"].numpy()

    D_q, D_scale = quantize_weight_pertensor(sd[f"blocks.{i}.D"].numpy())
    w["D_real"] = D_q * D_scale  # static per layer, precomputed once

    w.update(classic_recurrence_constants(sd, i, recurrence, s))
    return w


def prepare_quantized_model_classic(sd, ranges, hook_scales, d_inner, n_layers,
                                    h_width, recurrence):
    """Everything that depends only on the model/calibration, never on a
    specific clip. Call ONCE per run, not per clip.

    Scales are built BEFORE weights here (the selective version does the
    reverse), because the recurrence constants must be quantized against
    this layer's calibrated s['delta'] / s['B'] / s['C'] / s['A_bar'] /
    s['B_bar']."""
    h_clip = H_WIDTHS[h_width]

    scales = []
    for i in range(n_layers):
        s = {k.split(".", 1)[1]: v for k, v in ranges.items() if k.startswith(f"block{i}.")}
        for k, v in hook_scales.items():
            if k.startswith(f"block{i}."):
                s[k.split(".", 1)[1]] = v
        s["h_scale"] = ranges[f"block{i}.h"] * (127.0 / h_clip)
        scales.append(s)

    weights = [load_quantized_weights_classic(sd, i, d_inner, recurrence, scales[i])
               for i in range(n_layers)]

    luts_per_layer = []
    for i in range(n_layers):
        s = scales[i]
        lut_silu_conv, _ = build_lut(silu_fp32, s["conv_out"], s["u_post_conv_silu"])
        lut_silu_z, silu_z_scale = build_lut(silu_fp32, s["z_gate"], None)
        luts_per_layer.append({
            "silu_conv": lut_silu_conv,
            "silu_z": lut_silu_z,
            "silu_z_scale": silu_z_scale,
        })

    return weights, scales, luts_per_layer


# ---------------------------------------------------------------------------
# Per-frame arithmetic.
# ---------------------------------------------------------------------------

def quantized_block_step_classic(x_norm_q, x_norm_scale, w, s, luts, h_q, h_clip,
                                 conv_hist_q):
    """One block, one frame. Mirrors quantized_block_step line for line
    except that the x_proj/split/dt_proj/softplus section is replaced by
    w['A_bar_q'] / w['B_bar_q'] / w['C_q'], which are constants on this
    branch (see the module docstring on why hoisting them is exact)."""
    A_bar_q = w["A_bar_q"]
    B_bar_q = w["B_bar_q"]
    C_q = w["C_q"]

    u_raw_q, _ = quantized_linear(x_norm_q, x_norm_scale, w["in_proj_u_q"],
                                  w["in_proj_w_scale"], s["conv_out"])
    z_q, _ = quantized_linear(x_norm_q, x_norm_scale, w["in_proj_z_q"],
                              w["in_proj_w_scale"], s["z_gate"])

    # Depthwise conv, vectorized across all d_inner channels at once.
    taps_q = np.concatenate([conv_hist_q, u_raw_q[:, None]], axis=1)  # (d_inner, d_conv)
    acc = np.sum(w["conv_w_q"].astype(np.int64) * taps_q.astype(np.int64), axis=1)
    real = acc.astype(np.float64) * s["conv_out"] * w["conv_w_scale"] + w["conv_b_fp32"]
    conv_out_q = np.clip(np.round(real / s["conv_out"]), -127, 127).astype(np.int32)
    u_q = apply_lut(conv_out_q, luts["silu_conv"])

    new_conv_hist_q = np.roll(conv_hist_q, -1, axis=1)
    new_conv_hist_q[:, -1] = u_raw_q

    Ah_term = (A_bar_q.astype(np.int64) * h_q.astype(np.int64)).astype(np.float64) \
        * s["A_bar"] * s["h_scale"]
    Bu_term = (B_bar_q.astype(np.int64) * u_q[:, None].astype(np.int64)).astype(np.float64) \
        * s["B_bar"] * s["u_post_conv_silu"]
    new_h_real = Ah_term + Bu_term
    new_h_q = np.clip(np.round(new_h_real / s["h_scale"]), -h_clip, h_clip).astype(np.int32)

    y_scan_real = (new_h_q.astype(np.int64) @ C_q.astype(np.int64)).astype(np.float64) \
        * s["h_scale"] * s["C"]
    y_scan_q = np.clip(np.round(y_scan_real / s["y_scan"]), -127, 127).astype(np.int32)

    silu_z_q = apply_lut(z_q, luts["silu_z"])
    y_pre_gate_real = y_scan_q * s["y_scan"] + w["D_real"] * u_q * s["u_post_conv_silu"]
    y_gated_real = y_pre_gate_real * (silu_z_q * luts["silu_z_scale"])
    y_gated_q = np.clip(np.round(y_gated_real / s["y_gated"]), -127, 127).astype(np.int32)

    block_out_q, _ = quantized_linear(y_gated_q, s["y_gated"], w["out_proj_w_q"],
                                      w["out_proj_w_scale"], s["block_output"])

    return block_out_q, new_h_q, new_conv_hist_q


def quantized_forward_classic(x_fp32, sd, weights, scales, luts_per_layer, ranges,
                              d_model, d_inner, d_state, d_conv, n_layers, h_width):
    """weights/scales/luts_per_layer come from prepare_quantized_model_classic(),
    called ONCE by the caller. Returns the pooled embedding as a
    quantize-then-dequantize int8 round trip, matching the selective
    branch's quantized_forward exactly."""
    T = x_fp32.shape[0]
    h_clip = H_WIDTHS[h_width]
    final_norm_scale = ranges["final_norm_output"]

    h_state = [np.zeros((d_inner, d_state), dtype=np.int32) for _ in range(n_layers)]
    conv_hist = [np.zeros((d_inner, d_conv - 1), dtype=np.int32) for _ in range(n_layers)]

    pooled_sum = np.zeros(d_model)
    for t in range(T):
        x_real = x_fp32[t]
        for i in range(n_layers):
            s = scales[i]
            x_norm_real = rmsnorm_fp32(x_real, sd[f"norms.{i}.weight"].numpy())
            x_norm_q = quantize_int8(x_norm_real, s["norm_out"])

            block_out_q, h_state[i], conv_hist[i] = quantized_block_step_classic(
                x_norm_q, s["norm_out"], weights[i], s, luts_per_layer[i],
                h_state[i], h_clip, conv_hist[i])

            x_real = x_real + block_out_q * s["block_output"]

        normed_real = rmsnorm_fp32(x_real, sd["final_norm.weight"].numpy())
        normed_q = quantize_int8(normed_real, final_norm_scale)
        pooled_sum += normed_q * final_norm_scale

    pooled_real = pooled_sum / T
    pooled_q = quantize_int8(pooled_real, final_norm_scale)
    return pooled_q * final_norm_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--recurrence", choices=RECURRENCES, default="qab")
    parser.add_argument("--h-width", choices=list(H_WIDTHS), default="int8")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    if m["selective"]:
        raise ValueError("this simulator handles selective=False only; "
                         "use checks/smoke/quant/true_int8_sim.py for selective configs")
    if m["discretization"] != "euler":
        raise ValueError(f"classic true-int8 assumes euler; config says {m['discretization']!r}")

    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]

    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    model = SSMBackbone(**m)
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    fold = get_fold(args.held_out_case)
    mean, std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    ranges = load_ranges_scales(base_dir)
    hook_scales = calibrate_missing_ranges_classic(model, fold, mean, std, n_layers)

    weights, scales, luts_per_layer = prepare_quantized_model_classic(
        sd, ranges, hook_scales, d_inner, n_layers, args.h_width, args.recurrence)

    row = select_parity_clip(fold, clip_index=0)
    log_mel = np.load(row["cache_path"]).astype(np.float32)
    x = apply_normalization(log_mel, mean, std).astype(np.float32)

    with torch.no_grad():
        fp32_seq = model(torch.from_numpy(x).unsqueeze(0), mode="sequence", streaming=True)
        fp32_embedding = fp32_seq.mean(dim=1).squeeze(0).numpy()

    q_embedding = quantized_forward_classic(x, sd, weights, scales, luts_per_layer, ranges,
                                            d_model, d_inner, d_state, d_conv, n_layers,
                                            args.h_width)

    d = np.abs(q_embedding - fp32_embedding)
    print(f"\nrecurrence: {args.recurrence}   h_width: {args.h_width}")
    print(f"max_abs_err: {d.max():.6e}   mean_abs_err: {d.mean():.6e}")


if __name__ == "__main__":
    main()