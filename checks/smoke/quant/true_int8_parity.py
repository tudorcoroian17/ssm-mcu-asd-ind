"""
True-int8 (real int32-accumulate arithmetic) embedding parity check and
reference dump -- the true-int8 counterpart of combined_quant_parity.py.

Neither true_int8_sim.py's own __main__ nor true_int8_auc.py writes a
reference_embedding.npy, which is what check_backbone_parity.py needs to
compare against an on-device capture. This closes that gap: same parity
clip every other scheme uses (select_parity_clip, matching
PARITY_CLIP_STEM in check_backbone_parity.py), same fp32-vs-quantized
comparison true_int8_sim.py already prints, plus the dump.

Reuses true_int8_sim.py's calibration/weight-prep/forward functions
UNCHANGED -- this script only adds the dump step and the fp32 reference
computation (which true_int8_sim.py's own main() also does, via the
identical model(..., mode="sequence", streaming=True) call). If this
script's number ever disagrees with true_int8_sim.py's own printed
max_abs_err/mean_abs_err for the same clip/h-width, the bug is in this
script, never in a re-derivation of the quantized math.

WHAT THIS DOES NOT DO: verify the C port. This produces the Python-side
half of the parity check only. The other half is on hardware: flash the
true_int8_h8 (or true_int8_h16) build, feed this exact clip through
mcu/prepare_clip.py + audio_sender, capture the returned embedding as
<clip_stem>_embedding.npy, then run:
    python mcu/check_backbone_parity.py --scheme-dir mcu/deploy/case1/true_int8_h8
No amount of Python-side checking substitutes for that -- see the project's
established pattern (every prior scheme's parity number came from real
hardware, not a simulator).

Usage:
    python -m checks.smoke.quant.true_int8_parity \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --h-width int8 --dump-reference-dir mcu/deploy/case1/true_int8_h8
    python -m checks.smoke.quant.true_int8_parity \
        --config f4cd557b7e3b.yaml --held-out-case 1 --model-hash 16662b29beb3 \
        --h-width int16 --dump-reference-dir mcu/deploy/case1/true_int8_h16
"""
import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--held-out-case", type=int, required=True)
    parser.add_argument("--model-hash", required=True)
    parser.add_argument("--h-width", choices=list(H_WIDTHS), required=True)
    parser.add_argument("--dump-reference-dir", required=True,
                        help="mcu/deploy/case<N>/true_int8_h8|h16/ -- writes "
                             "reference_embedding.npy there")
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

    print("Calibrating missing ranges (dt_proj/conv/norm hooks)...")
    hook_scales = calibrate_missing_ranges(model, fold, mean, std, n_layers, dt_rank)

    print(f"Preparing quantized weights, scales, and LUTs [h={args.h_width}]...")
    weights, scales, luts_per_layer = prepare_quantized_model(
        sd, ranges, hook_scales, d_inner, n_layers, args.h_width)

    # Same parity clip as EVERY other scheme -- check_backbone_parity.py's
    # PARITY_CLIP_STEM assumes this, and an on-device capture from a
    # different clip would silently compare against the wrong reference.
    row = select_parity_clip(fold, clip_index=0)
    log_mel = np.load(row["cache_path"]).astype(np.float32)
    x = apply_normalization(log_mel, mean, std).astype(np.float32)

    with torch.no_grad():
        fp32_seq = model(torch.from_numpy(x).unsqueeze(0), mode="sequence", streaming=True)
        fp32_embedding = fp32_seq.mean(dim=1).squeeze(0).numpy()

    q_embedding = quantized_forward(x, sd, weights, scales, luts_per_layer, ranges,
                                    d_model, d_inner, d_state, d_conv, n_layers, args.h_width)

    d = np.abs(q_embedding - fp32_embedding)
    print(f"\nclip: {row['cache_path']}")
    print(f"h_width: {args.h_width}")
    print(f"max_abs_err (fp32 vs true-int8): {d.max():.6e}   "
          f"mean_abs_err: {d.mean():.6e}")

    out_dir = Path(args.dump_reference_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reference_embedding.npy"
    np.save(out_path, q_embedding)
    print(f"\nWrote true-int8 [{args.h_width}] reference embedding to {out_path}")
    print("Next: flash the matching true_int8_h8/h16 build, run this same clip "
          "through audio_sender, then:")
    print(f"    python mcu/check_backbone_parity.py --scheme-dir {out_dir}")


if __name__ == "__main__":
    main()