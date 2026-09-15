import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=str, required=True)
    parser.add_argument('--freq-ms', type=int, required=True)
    parser.add_argument('--frames', type=int, default=344)
    args = parser.parse_args()

    timing_file_path = Path("/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/test_harness/output/cycle_sweep_summary.csv")
    timing_df = pd.read_csv(timing_file_path)
    board_list = ['stm32-nucleo-h7s3l8', 'esp32', 'arduino-uno-rp2040']

    if args.board not in board_list:
        raise ValueError(f'Board not supported. Must be one of {board_list}')

    out_dir = PROJECT_ROOT / 'mcu' / 'eval' / 'online' / args.board

    # Millisecond conversions
    cycles_per_ms = args.freq_ms * 1000
    timing_df["feature_ms"] = timing_df["mean_feature_cycles"] / cycles_per_ms
    timing_df["normalize_ms"] = timing_df["mean_normalize_cycles"] / cycles_per_ms
    timing_df["backbone_ms"] = timing_df["mean_backbone_cycles"] / cycles_per_ms
    timing_df["head_ms"] = timing_df["mean_head_cycles"] / cycles_per_ms

    # Per-frame conversions
    timing_df["feature_ms_perframe"] = timing_df["feature_ms"] / args.frames
    timing_df["normalize_ms_perframe"] = timing_df["normalize_ms"] / args.frames
    timing_df["backbone_ms_perframe"] = timing_df["backbone_ms"] / args.frames
    timing_df["head_ms_perframe"] = timing_df["head_ms"] / args.frames

    timing_df.to_csv(out_dir / "timing_results.csv", index=False)

if __name__ == '__main__':
    main()