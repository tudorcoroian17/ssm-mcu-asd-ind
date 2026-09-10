"""
True int8 quantized-inference simulator -- int32-accumulate, direct
fixed-point discretize math, LUT-based SiLU/softplus, swept h storage
width. NOT fake-quant (weight_quant_parity.py / activation_quant_parity.py
dequantize immediately and compute fp32); every matmul, the depthwise
conv, and the recurrence here genuinely accumulate in int32 and requantize
to int8/int16, matching what a hand-written CMSIS-NN-style C kernel would
do.

Motivation: findings/520-531 validate that int8 ROUNDING NOISE is
tolerable. They say nothing about chained integer arithmetic -- int32
accumulate, explicit rescale, repeated every op, hundreds of times per
clip through the recurrence. This answers that in Python before any C is
written.

CALIBRATION: ranges.json (findings/150) has B, C, A_bar, B_bar, h, delta
(post-softplus), u_post_conv_silu (post-SiLU), z_gate, y_scan, y_gated,
block_output, final_norm_output. Missing, and calibrated here via forward
hooks on existing submodules (no ssm_block.py edits): dt_proj's INPUT
(delta_low, needed to requantize x_proj's split output), dt_proj's OUTPUT
(delta_raw, the softplus LUT's input range), conv's OUTPUT (the SiLU-at-
conv LUT's input range), and each block's RMSNorm output (x_norm feeding
into that block).

DELIBERATE EXCEPTIONS -- not fully quantized, on purpose, not by oversight:
  - RMSNorm's arithmetic (sum of squares, rsqrt) stays fp32 -- no clean
    int8 form, same category of problem as an int8 FFT. Its OUTPUT is
    still quantized to int8 before the next block.
  - Log-mel extraction is untouched fp32, same reasoning.
  - Bias terms are added in real (dequantized) units at each op's rescale
    point, not separately quantized to int32. Biases are small
    (findings/521); this trades a small amount of fidelity for real
    simplicity, and is worth revisiting only if it turns out to matter.

Loads existing checkpoints, does not train.

Usage:
    python -m checks.smoke.quant.true_int8_sim \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --h-width int8
    python -m checks.smoke.quant.true_int8_sim \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --h-width int16
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

H_WIDTHS = {"int8": 127, "int16": 32767}


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def quantize_int8(x_fp32, scale):
    return np.clip(np.round(x_fp32 / scale), -127, 127).astype(np.int32)


def quantize_weight_pertensor(w_fp32):
    """Same math as weight_quant_parity.py's quantize_dequantize, but
    returns raw int8 values (for real arithmetic) instead of the
    dequantized float (for accuracy-only checks)."""
    max_abs = float(np.max(np.abs(w_fp32)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1e-12
    q = np.clip(np.round(w_fp32 / scale), -127, 127).astype(np.int32)
    return q, scale


def quantized_linear(x_q, x_scale, w_q, w_scale, out_scale, bias_fp32=None):
    """x_q: (in,) int32 in [-127,127]. w_q: (out, in) int32. Returns
    (out,) int32, requantized to out_scale. int64 accumulation avoids any
    overflow risk (max product 127*127=16129, times up to a few hundred
    reduction terms -- nowhere near int64's range, just extra headroom
    over the int32 a real kernel would use)."""
    acc = w_q.astype(np.int64) @ x_q.astype(np.int64)
    real = acc.astype(np.float64) * x_scale * w_scale
    if bias_fp32 is not None:
        real = real + bias_fp32
    return np.clip(np.round(real / out_scale), -127, 127).astype(np.int32), real


def build_lut(fn, in_scale, out_scale=None):
    """256 entries, index i+128 holds f(i) for i in [-128,127]. If
    out_scale is None (no independent calibration exists for this
    function's output -- true for silu-at-z-gate, which is never itself
    recorded), derive it from the function's own range over the input
    domain, exactly as evaluated here."""
    raw = np.array([fn(i * in_scale) for i in range(-128, 128)])
    if out_scale is None:
        max_abs = float(np.max(np.abs(raw)))
        out_scale = max_abs / 127.0 if max_abs > 0.0 else 1e-12
    lut = np.clip(np.round(raw / out_scale), -127, 127).astype(np.int32)
    return lut, out_scale


def apply_lut(x_q, lut):
    return lut[np.asarray(x_q, dtype=np.int64) + 128]


def softplus_fp32(x):
    return x if x > 20.0 else np.log1p(np.exp(x))


def silu_fp32(x):
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Calibration -- ranges.json plus four hook-derived ranges it lacks.
# ---------------------------------------------------------------------------

def calibrate_missing_ranges(model, fold, mean, std, n_layers, dt_rank, n_clips=8):
    captured = {}
    for i in range(n_layers):
        for key in ("x_proj_delta_low_out", "dt_proj_out", "conv_out", "norm_out"):
            captured[f"block{i}.{key}"] = []

    hooks = []
    for i, block in enumerate(model.blocks):
        def x_proj_hook(module, inp, out, i=i):
            captured[f"block{i}.x_proj_delta_low_out"].append(
                out[..., :dt_rank].detach().abs().max().item())
        def dt_proj_hook(module, inp, out, i=i):
            captured[f"block{i}.dt_proj_out"].append(out.detach().abs().max().item())
        def conv_hook(module, inp, out, i=i):
            captured[f"block{i}.conv_out"].append(out.detach().abs().max().item())
        def norm_hook(module, inp, out, i=i):
            captured[f"block{i}.norm_out"].append(out.detach().abs().max().item())

        hooks.append(block.x_proj.register_forward_hook(x_proj_hook))
        hooks.append(block.dt_proj.register_forward_hook(dt_proj_hook))
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
# Weight loading -- everything quantized, per-tensor.
# ---------------------------------------------------------------------------

def load_quantized_weights(sd, i, d_inner):
    w = {}
    for name, key in [
        ("in_proj_w", f"blocks.{i}.in_proj.weight"),
        ("conv_w", f"blocks.{i}.conv.weight"),
        ("x_proj_w", f"blocks.{i}.x_proj.weight"),
        ("dt_proj_w", f"blocks.{i}.dt_proj.weight"),
        ("out_proj_w", f"blocks.{i}.out_proj.weight"),
        ("norm_w", f"norms.{i}.weight"),
    ]:
        q, scale = quantize_weight_pertensor(sd[key].numpy())
        w[f"{name}_q"], w[f"{name}_scale"] = q, scale

    w["conv_w_q"] = w["conv_w_q"].reshape(d_inner, -1)  # (d_inner, 1, d_conv) -> (d_inner, d_conv)
    w["in_proj_u_q"] = w["in_proj_w_q"][:d_inner]
    w["in_proj_z_q"] = w["in_proj_w_q"][d_inner:]

    w["conv_b_fp32"] = sd[f"blocks.{i}.conv.bias"].numpy()
    w["dt_proj_b_fp32"] = sd[f"blocks.{i}.dt_proj.bias"].numpy()

    A_log_q, A_log_scale = quantize_weight_pertensor(sd[f"blocks.{i}.A_log"].numpy())
    w["A_real"] = -np.exp(A_log_q * A_log_scale)  # static per layer, precomputed once
    D_q, D_scale = quantize_weight_pertensor(sd[f"blocks.{i}.D"].numpy())
    w["D_real"] = D_q * D_scale  # static per layer, precomputed once

    return w


def prepare_quantized_model(sd, ranges, hook_scales, d_inner, n_layers, h_width):
    """Everything that depends only on the model/calibration, never on a
    specific clip: quantized weights, per-layer scale dicts, LUTs, and h's
    scale (fixed once h_width is chosen). Call this ONCE per run, not per
    clip -- weight quantization and LUT construction are wasted work if
    repeated per clip, and for an AUC check over hundreds of clips that
    waste is the difference between minutes and hours."""
    h_clip = H_WIDTHS[h_width]
    weights = [load_quantized_weights(sd, i, d_inner) for i in range(n_layers)]

    scales = []
    for i in range(n_layers):
        s = {k.split(".", 1)[1]: v for k, v in ranges.items() if k.startswith(f"block{i}.")}
        for k, v in hook_scales.items():
            if k.startswith(f"block{i}."):
                s[k.split(".", 1)[1]] = v
        s["h_scale"] = ranges[f"block{i}.h"] * (127.0 / h_clip)
        scales.append(s)

    luts_per_layer = []
    for i in range(n_layers):
        s = scales[i]
        lut_softplus, _ = build_lut(softplus_fp32, s["dt_proj_out"], s["delta"])
        lut_silu_conv, _ = build_lut(silu_fp32, s["conv_out"], s["u_post_conv_silu"])
        lut_silu_z, silu_z_scale = build_lut(silu_fp32, s["z_gate"], None)
        luts_per_layer.append({
            "softplus": lut_softplus, "silu_conv": lut_silu_conv,
            "silu_z": lut_silu_z, "silu_z_scale": silu_z_scale,
        })

    return weights, scales, luts_per_layer


def quantized_block_step(x_norm_q, x_norm_scale, w, s, luts, h_q, h_clip, conv_hist_q):
    d_inner, d_state = w["A_real"].shape

    u_raw_q, _ = quantized_linear(x_norm_q, x_norm_scale, w["in_proj_u_q"],
                                  w["in_proj_w_scale"], s["conv_out"])
    z_q, _ = quantized_linear(x_norm_q, x_norm_scale, w["in_proj_z_q"],
                              w["in_proj_w_scale"], s["z_gate"])

    # Depthwise conv, vectorized across all d_inner channels at once --
    # same per-channel dot product as before, just done as one numpy call
    # instead of a 128-iteration Python loop. Identical arithmetic.
    taps_q = np.concatenate([conv_hist_q, u_raw_q[:, None]], axis=1)  # (d_inner, d_conv)
    acc = np.sum(w["conv_w_q"].astype(np.int64) * taps_q.astype(np.int64), axis=1)
    real = acc.astype(np.float64) * s["conv_out"] * w["conv_w_scale"] + w["conv_b_fp32"]
    conv_out_q = np.clip(np.round(real / s["conv_out"]), -127, 127).astype(np.int32)
    u_q = apply_lut(conv_out_q, luts["silu_conv"])

    new_conv_hist_q = np.roll(conv_hist_q, -1, axis=1)
    new_conv_hist_q[:, -1] = u_raw_q

    x_dbl_real = w["x_proj_w_q"].astype(np.int64) @ u_q.astype(np.int64)
    x_dbl_real = x_dbl_real.astype(np.float64) * s["u_post_conv_silu"] * w["x_proj_w_scale"]
    dt_rank = w["dt_proj_w_q"].shape[1]
    delta_low_real, B_real, C_real = np.split(x_dbl_real, [dt_rank, dt_rank + d_state])
    delta_low_q = np.clip(np.round(delta_low_real / s["x_proj_delta_low_out"]), -127, 127).astype(np.int32)
    B_q = np.clip(np.round(B_real / s["B"]), -127, 127).astype(np.int32)
    C_q = np.clip(np.round(C_real / s["C"]), -127, 127).astype(np.int32)

    delta_raw_q, _ = quantized_linear(delta_low_q, s["x_proj_delta_low_out"],
                                      w["dt_proj_w_q"], w["dt_proj_w_scale"],
                                      s["dt_proj_out"], bias_fp32=w["dt_proj_b_fp32"])
    delta_q = apply_lut(delta_raw_q, luts["softplus"])

    delta_real = delta_q * s["delta"]
    deltaA_real = delta_real[:, None] * w["A_real"]
    deltaA_clamped = np.maximum(deltaA_real, -1.9)
    A_bar_real = 1.0 + deltaA_clamped
    A_bar_q = np.clip(np.round(A_bar_real / s["A_bar"]), -127, 127).astype(np.int32)

    B_bar_real = (delta_q[:, None].astype(np.int64) * B_q[None, :].astype(np.int64)) \
        .astype(np.float64) * s["delta"] * s["B"]
    B_bar_q = np.clip(np.round(B_bar_real / s["B_bar"]), -127, 127).astype(np.int32)

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


def rmsnorm_fp32(x_real, weight_fp32, eps=1e-5):
    ss = np.mean(x_real ** 2)
    return x_real / np.sqrt(ss + eps) * weight_fp32


def quantized_forward(x_fp32, sd, weights, scales, luts_per_layer, ranges,
                      d_model, d_inner, d_state, d_conv, n_layers, h_width):
    """weights/scales/luts_per_layer come from prepare_quantized_model(),
    called ONCE by the caller and reused across every clip -- not rebuilt
    here. Returns the pooled embedding as a quantize-then-dequantize int8
    round-trip (see module docstring / the fix noted where this was added):
    the actual returned values are what an int8-embedding device would
    produce, not a silently-fp32 pooled sum."""
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

            block_out_q, h_state[i], conv_hist[i] = quantized_block_step(
                x_norm_q, s["norm_out"], weights[i], s, luts_per_layer[i],
                h_state[i], h_clip, conv_hist[i])

            x_real = x_real + block_out_q * s["block_output"]

        normed_real = rmsnorm_fp32(x_real, sd["final_norm.weight"].numpy())
        # normed is int8 between frames (matches real storage); the
        # pooling ACCUMULATOR stays wide (real units), same "accumulate
        # wide, requantize at boundaries" principle used everywhere else.
        normed_q = quantize_int8(normed_real, final_norm_scale)
        pooled_sum += normed_q * final_norm_scale

    pooled_real = pooled_sum / T
    # Final requantization: the embedding genuinely leaves this function as
    # an int8 round-trip, not fp32 -- this was missing before.
    pooled_q = quantize_int8(pooled_real, final_norm_scale)
    return pooled_q * final_norm_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--h-width", choices=list(H_WIDTHS), default="int8")
    args = parser.parse_args()

    cfg = load_config_by_name(args.config)
    m = cfg["model"]
    d_model, d_state = m["d_model"], m["d_state"]
    d_inner, d_conv, n_layers = m["expand"] * d_model, m["d_conv"], m["n_layers"]
    dt_rank = max(1, d_model // 16)

    base_dir = PROJECT_ROOT / "runs" / f"case{args.held_out_case}" / args.model_hash
    model = SSMBackbone(**m)
    ckpt = torch.load(base_dir / "ckpt.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    fold = get_fold(args.held_out_case)
    mean, std, _ = compute_normalization_stats(fold["train"]["cache_path"], m["n_mels"])
    ranges = load_ranges_scales(base_dir)
    hook_scales = calibrate_missing_ranges(model, fold, mean, std, n_layers, dt_rank)

    weights, scales, luts_per_layer = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, args.h_width)

    row = select_parity_clip(fold, clip_index=0)
    log_mel = np.load(row["cache_path"]).astype(np.float32)
    x = apply_normalization(log_mel, mean, std).astype(np.float32)

    with torch.no_grad():
        fp32_seq = model(torch.from_numpy(x).unsqueeze(0), mode="sequence", streaming=True)
        fp32_embedding = fp32_seq.mean(dim=1).squeeze(0).numpy()

    q_embedding = quantized_forward(x, sd, weights, scales, luts_per_layer, ranges,
                                    d_model, d_inner, d_state, d_conv, n_layers, args.h_width)

    d = np.abs(q_embedding - fp32_embedding)
    print(f"\nh_width: {args.h_width}")
    print(f"max_abs_err: {d.max():.6e}   mean_abs_err: {d.mean():.6e}")


if __name__ == "__main__":
    main()