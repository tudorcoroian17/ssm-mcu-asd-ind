"""
Rolls up power_decode.py's per-clip frame CSVs into one master CSV,
mirroring how collect_timing_results.py rolls up cycle_sweep.py's
output. No hardware, no live PPK2 -- purely reads already-decoded
CSVs, so this runs from WSL like the rest of mcu/eval/.

For each (case, model_hash, setup_id, clip):
  - a time-weighted mean current per section (feature/backbone/total) --
    sum(current_i * delta_t_i) / sum(delta_t_i), the energy-consistent
    way to average currents over windows of slightly unequal duration,
    not a naive mean of per-pulse means
  - total time spent in that section for the whole clip (sum of
    delta_t across every hop, plus head's own window for "total")
  - the DWT-derived cycle counts from that same clip's timing.csv
    (power_sweep.py already wrote this, read here off the Windows
    drive the same way collect_timing_results.py already reads
    cycle_sweep_summary.csv) -- so the PPK2-derived total sits right
    next to the firmware's own cycle-counted total for a sanity check,
    without a separate script.

Usage:
    python collect_power_results.py --board stm32-nucleo-h7s3l8
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

MCU_CLOCK_HZ = 600_000_000  # SystemCoreClock, confirmed from the .ioc

WINDOWS_OUTPUT_POWER_ROOT = Path(
    "/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/test_harness/output_power")


def weighted_mean_current(df, prefix):
    current = df[f"{prefix}_current_ua"]
    delta_t = df[f"{prefix}_delta_t_s"]
    valid = current.notna() & delta_t.notna()
    total_t = delta_t[valid].sum()
    if total_t == 0 or not valid.any():
        return np.nan, 0.0
    return float((current[valid] * delta_t[valid]).sum() / total_t), float(total_t)


def dwt_row(case, model_hash, setup_id, clip_name):
    """Best-effort: the matching row in power_sweep.py's timing.csv. A
    missing file or row returns all-NaN rather than raising -- one
    absent cross-check shouldn't stop the rest of the master CSV."""
    empty = {"feature_cycles_dwt": np.nan, "feature_time_dwt_s": np.nan,
            "normalize_cycles_dwt": np.nan, "backbone_cycles_dwt": np.nan,
            "backbone_time_dwt_s": np.nan, "head_cycles_dwt": np.nan,
            "total_time_dwt_s": np.nan}
    timing_path = WINDOWS_OUTPUT_POWER_ROOT / case / model_hash / setup_id / "timing.csv"
    if not timing_path.exists():
        return empty
    timing_df = pd.read_csv(timing_path)
    match = timing_df[timing_df["clip_name"] == clip_name]
    if match.empty:
        return empty
    row = match.iloc[-1]
    feature_c, normalize_c = float(row["feature_cycles"]), float(row["normalize_cycles"])
    backbone_c, head_c = float(row["backbone_cycles"]), float(row["head_cycles"])
    return {"feature_cycles_dwt": feature_c, "feature_time_dwt_s": feature_c / MCU_CLOCK_HZ,
           "normalize_cycles_dwt": normalize_c, "backbone_cycles_dwt": backbone_c,
           "backbone_time_dwt_s": backbone_c / MCU_CLOCK_HZ, "head_cycles_dwt": head_c,
           "total_time_dwt_s": (feature_c + normalize_c + backbone_c + head_c) / MCU_CLOCK_HZ}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", required=True)
    args = parser.parse_args()

    deploy_root = PROJECT_ROOT / "mcu" / "deploy"
    frame_paths = sorted(deploy_root.glob(f"case*/*/setups/*/online/{args.board}/power/*_frames.csv"))
    if not frame_paths:
        raise SystemExit(f"No frame CSVs found under {deploy_root}/case*/*/setups/*/online/"
                         f"{args.board}/power/ -- run power_decode.py first.")

    rows = []
    for frame_path in frame_paths:
        # .../case<N>/<model_hash>/setups/<setup_id>/online/<board>/power/<clip>_frames.csv
        setup_id = frame_path.parents[3].name
        model_hash = frame_path.parents[5].name
        case = frame_path.parents[6].name

        df = pd.read_csv(frame_path)
        clip_name, role = df["clip_name"].iloc[0], df["role"].iloc[0]
        n_hops = int(df["phase"].eq("hop").sum())

        feature_mean, feature_total_t = weighted_mean_current(df, "feature")
        backbone_mean, backbone_total_t = weighted_mean_current(df, "backbone")
        total_mean, total_total_t = weighted_mean_current(df, "total")

        row = {"case": case, "model_hash": model_hash, "setup_id": setup_id,
              "clip_name": clip_name, "role": role, "n_hops": n_hops,
              "feature_current_ua_mean": feature_mean, "feature_delta_t_s_total": feature_total_t,
              "backbone_current_ua_mean": backbone_mean, "backbone_delta_t_s_total": backbone_total_t,
              "total_current_ua_mean": total_mean, "total_delta_t_s_total": total_total_t}
        row.update(dwt_row(case, model_hash, setup_id, clip_name))
        rows.append(row)

    out_dir = PROJECT_ROOT / "mcu" / "eval" / "online" / args.board / 'power'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "power_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_setups = len(set((r["case"], r["model_hash"], r["setup_id"]) for r in rows))
    print(f"Wrote {out_path} ({len(rows)} clip row(s) across {n_setups} setup(s))")


if __name__ == "__main__":
    main()