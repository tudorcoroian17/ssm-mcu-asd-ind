"""
Converts compute_current_results.py's current (uA) measurements into
energy (mJ) and power (mW), for both the master CSV and the per-clip
frame CSVs. Pure arithmetic on already-decoded CSVs -- no hardware, no
live PPK2 -- so this runs from WSL like the rest of mcu/eval/.

power_mw = current_ua * voltage_mv * 1e-6
energy_mj = power_mw * delta_t_s   (mW * s = mJ, dimensionally exact)

Voltage is a REQUIRED argument, not read from anywhere automatically:
set_source_voltage() only calibrates PPK2's own ADC math (confirmed
from ppk2_api's own source), it is not a guarantee of the real
VDD_MCU rail voltage -- use whatever you measured on the bench.

Usage:
    python compute_energy_power_results.py --voltage-mv 3300
    python compute_energy_power_results.py --voltage-mv 3300 --input master --metric energy
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

SECTIONS = ("feature", "backbone", "total")


def add_power_and_energy(df, voltage_mv, delta_t_suffix, current_suffix, id_cols):
    """Returns (power_df, energy_df), each carrying id_cols plus, per
    section, the kept delta_t column and the one new derived column.
    current_ua is deliberately not carried into either output -- it's
    one multiply-by-voltage away in the file that already owns it."""
    power_df = df[list(id_cols)].copy()
    energy_df = df[list(id_cols)].copy()
    for section in SECTIONS:
        delta_t_col = f"{section}_{delta_t_suffix}"
        current_col = f"{section}_{current_suffix}"
        power_mw = df[current_col] * voltage_mv * 1e-6

        power_df[delta_t_col] = df[delta_t_col]
        power_df[f"{section}_power_mw"] = power_mw

        energy_df[delta_t_col] = df[delta_t_col]
        energy_df[f"{section}_energy_mj"] = power_mw * df[delta_t_col]
    return power_df, energy_df


def process_master(voltage_mv, metrics, board):
    master_dir = PROJECT_ROOT / "mcu" / "eval" / "online" / board / "pec"
    current_path = master_dir / "current_results.csv"
    if not current_path.exists():
        raise SystemExit(f"{current_path} not found -- run compute_current_results.py first.")

    df = pd.read_csv(current_path)
    id_cols = ["case", "model_hash", "setup_id", "clip_name", "role", "n_hops"]
    power_df, energy_df = add_power_and_energy(
        df, voltage_mv, delta_t_suffix="delta_t_s_total", current_suffix="current_ua_mean",
        id_cols=id_cols)

    if "power" in metrics:
        out_path = master_dir / "power_results.csv"
        power_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(power_df)} row(s))")
    if "energy" in metrics:
        out_path = master_dir / "energy_results.csv"
        energy_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(energy_df)} row(s))")


def process_individual(voltage_mv, metrics, board):
    deploy_root = PROJECT_ROOT / "mcu" / "deploy"
    frame_paths = sorted(deploy_root.glob(f"case*/*/setups/*/online/{board}/current/*_frames.csv"))
    if not frame_paths:
        raise SystemExit(f"No frame CSVs found under {deploy_root}/case*/*/setups/*/online/"
                         f"{board}/current/ -- run power_decode.py first.")

    id_cols = ["clip_name", "role", "frame_index", "phase"]
    n_written = {"energy": 0, "power": 0}

    for frame_path in frame_paths:
        df = pd.read_csv(frame_path)
        power_df, energy_df = add_power_and_energy(
            df, voltage_mv, delta_t_suffix="delta_t_s", current_suffix="current_ua", id_cols=id_cols)

        board_dir = frame_path.parents[1]  # .../online/<board>
        if "power" in metrics:
            out_path = board_dir / "power" / frame_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            power_df.to_csv(out_path, index=False)
            n_written["power"] += 1
        if "energy" in metrics:
            out_path = board_dir / "energy" / frame_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            energy_df.to_csv(out_path, index=False)
            n_written["energy"] += 1

    for metric, count in n_written.items():
        if count:
            print(f"Wrote {count} per-clip {metric} CSV(s) under "
                 f"case*/*/setups/*/online/{board}/{metric}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voltage-mv", type=int, required=True,
                        help="measured VDD_MCU rail voltage in mV, e.g. 3300")
    parser.add_argument("--input", choices=["master", "individual", "both"], default="both")
    parser.add_argument("--metric", choices=["energy", "power", "both"], default="both")
    parser.add_argument("--board", type=str, required=True)
    args = parser.parse_args()

    board_list = ['stm32-nucleo-h7s3l8', 'esp32', 'arduino-nano-rp2040-connect-mbed',
                  'arduino-nano-rp2040-connect-pico', 'arduino-nano-rp2040-connect-pico-q15']
    if args.board not in board_list:
        raise ValueError(f'Board not supported. Must be one of {board_list}')

    metrics = ["energy", "power"] if args.metric == "both" else [args.metric]

    if args.input in ("master", "both"):
        process_master(args.voltage_mv, metrics, args.board)
    if args.input in ("individual", "both"):
        process_individual(args.voltage_mv, metrics, args.board)


if __name__ == "__main__":
    main()