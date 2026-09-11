"""
ONE-OFF DIAGNOSTIC (delete after use), superseding the previous two
versions of this file.

Ruled out so far:
  - rounding convention (round-half-away-from-zero vs np.round): zero
    difference -- no .5 ties occurred anywhere in this clip.
  - float32 vs float64 precision: zero difference -- bit-identical output.

New hypothesis: the Python-side reference has been feeding quantized_forward
the CACHED librosa log-mel (cache_path), but the device never sees that --
it computes its own log-mel on-device via CMSIS-DSP, and Phase 3 already
established and accepted a real discrepancy there (max abs diff 0.0617
against a derived tolerance of 0.2627 -- findings around
05_phase_4_backbone_port.md / the feature-pipeline parity check). fp32
absorbs an input wobble that size smoothly; true int8's recurrent,
discrete-decision-per-frame arithmetic can let it flip a rounding boundary
that then persists in h. This feeds the DEVICE'S OWN captured log-mel
(already on disk, no reflash) through the same, already-validated
quantized_forward instead of the cache file, and compares both the
resulting embedding AND the raw log-mel itself against the cached version,
to see if the magnitude of feature-pipeline drift alone explains the
0.17 embedding-level gap.

Usage:
    python to_delete.py --config f4cd557b7e3b.yaml --held-out-case 1 \
        --model-hash 16662b29beb3 --h-width int8 \
        --device-logmel "/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/audio_sender/clip_samples_logmel_parity/output/1200010003_ToyCar_case2_normal_IND_ch1_0003.npy"
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
    H_WIDTHS, calibrate_missing_ranges, prepare_quantized_model, quantized_forward,
)
from mcu.check_backbone_parity import EMBEDDING_DIR, PARITY_CLIP_STEM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--h-width", choices=list(H_WIDTHS), required=True)
    parser.add_argument("--device-logmel", required=True,
                        help="the device's own captured log-mel for the "
                             "parity clip (findings 322-325's debug capture)")
    args = parser.parse_args()

    on_device_embedding = np.load(EMBEDDING_DIR / f"{PARITY_CLIP_STEM}_embedding.npy")

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
    cached_log_mel = np.load(row["cache_path"]).astype(np.float32)
    device_log_mel = np.load(args.device_logmel).astype(np.float32)

    print(f"clip: {row['cache_path']}")
    print(f"cached (librosa) log-mel shape: {cached_log_mel.shape}")
    print(f"device (CMSIS-DSP) log-mel shape: {device_log_mel.shape}")

    if cached_log_mel.shape != device_log_mel.shape:
        n = min(cached_log_mel.shape[0], device_log_mel.shape[0])
        print(f"WARNING: shape mismatch -- truncating both to {n} frames "
              f"for comparison. A frame-count mismatch is itself worth "
              f"noting separately if it persists.")
        cached_log_mel = cached_log_mel[:n]
        device_log_mel = device_log_mel[:n]

    logmel_diff = np.abs(cached_log_mel - device_log_mel)
    print(f"\nlog-mel diff, cached vs device (should be in the same "
          f"ballpark as Phase 3's own 0.0617 max / tolerance 0.2627):")
    print(f"  max_abs_diff={logmel_diff.max():.6e}  mean_abs_diff={logmel_diff.mean():.6e}")

    x_cached = apply_normalization(cached_log_mel, mean, std).astype(np.float32)
    x_device = apply_normalization(device_log_mel, mean, std).astype(np.float32)

    ref_from_cached = quantized_forward(x_cached, sd, weights, scales, luts_per_layer, ranges,
                                        d_model, d_inner, d_state, d_conv, n_layers, args.h_width)
    ref_from_device_logmel = quantized_forward(x_device, sd, weights, scales, luts_per_layer, ranges,
                                               d_model, d_inner, d_state, d_conv, n_layers, args.h_width)

    d_cached_vs_device_embedding = np.abs(on_device_embedding - ref_from_cached)
    d_new = np.abs(on_device_embedding - ref_from_device_logmel)

    print(f"\n[before] reference from CACHED log-mel vs on-device embedding "
          f"(matches earlier runs):")
    print(f"  max_abs_err={d_cached_vs_device_embedding.max():.6e}  "
          f"mean_abs_err={d_cached_vs_device_embedding.mean():.6e}")
    print(f"[after]  reference from the DEVICE'S OWN log-mel vs on-device "
          f"embedding:")
    print(f"  max_abs_err={d_new.max():.6e}  mean_abs_err={d_new.mean():.6e}")


if __name__ == "__main__":
    main()