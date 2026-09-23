import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

# The Arduino Nano RP2040 Connect is two boards here, one per Arduino core.
# The core is part of the board name because it changes the compiler and the
# float library, so results from the two builds must never mix.
RP2040_MBED = 'arduino-nano-rp2040-connect-mbed'
RP2040_PICO = 'arduino-nano-rp2040-connect-pico'
RP2040_PICO_Q15 = 'arduino-nano-rp2040-connect-pico-q15'

BOARD_LIST = ['stm32-nucleo-h7s3l8', 'esp32', RP2040_MBED, RP2040_PICO, RP2040_PICO_Q15]

BOARD_TIMING_UNITS = {
    'stm32-nucleo-h7s3l8': 'cycles',
    'esp32': 'cycles',
    RP2040_MBED: 'microseconds',
    RP2040_PICO: 'microseconds',
    RP2040_PICO_Q15: 'microseconds',
}

RP2040_TIMING_COLUMNS = {
    'feature': 'mean_feature_us', 'normalize': 'mean_normalize_us',
    'backbone': 'mean_backbone_us', 'head': 'mean_head_us',
}

_TEST_HARNESS = Path("/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/test_harness")

BOARD_TIMING_SOURCE = {
    'stm32-nucleo-h7s3l8': {
        'path': _TEST_HARNESS / "output" / "cycle_sweep_summary.csv",
        'columns': {
            'feature': 'mean_feature_cycles', 'normalize': 'mean_normalize_cycles',
            'backbone': 'mean_backbone_cycles', 'head': 'mean_head_cycles',
        },
    },
    RP2040_MBED: {
        'path': _TEST_HARNESS / "output_rp2040_naive" / "cycle_sweep_rp2040_summary.csv",
        'columns': RP2040_TIMING_COLUMNS,
    },
    RP2040_PICO: {
        'path': _TEST_HARNESS / "output_rp2040_pico" / "cycle_sweep_rp2040_summary.csv",
        'columns': RP2040_TIMING_COLUMNS,
    },
    RP2040_PICO_Q15: {
        'path': _TEST_HARNESS / "output_rp2040_pico_q15" / "cycle_sweep_rp2040_summary.csv",
        'columns': RP2040_TIMING_COLUMNS,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=str, required=True)
    parser.add_argument('--freq-mhz', type=int, default=None,
                        help='Required for cycle-based boards; ignored for '
                             'boards whose firmware already reports microseconds.')
    parser.add_argument('--frames', type=int, default=344)
    parser.add_argument('--timing-file', type=Path, default=None,
                        help='Overrides the per-board default summary CSV path.')
    args = parser.parse_args()

    if args.board not in BOARD_LIST:
        raise ValueError(f'Board not supported. Must be one of {BOARD_LIST}')

    source = BOARD_TIMING_SOURCE[args.board]
    timing_file_path = args.timing_file or source['path']
    timing_df = pd.read_csv(timing_file_path)
    cols = source['columns']

    unit = BOARD_TIMING_UNITS[args.board]
    if unit == 'cycles':
        if args.freq_mhz is None:
            raise ValueError(f'--freq-mhz is required for board {args.board!r} '
                             f'(timing unit: cycles).')
        cycles_per_ms = args.freq_mhz * 1000
        for stage, col in cols.items():
            timing_df[f'{stage}_ms'] = timing_df[col] / cycles_per_ms
    elif unit == 'microseconds':
        for stage, col in cols.items():
            timing_df[f'{stage}_ms'] = timing_df[col] / 1000.0
    else:
        raise ValueError(f'Unknown timing unit {unit!r} for board {args.board!r}')

    for stage in cols:
        timing_df[f'{stage}_ms_perframe'] = timing_df[f'{stage}_ms'] / args.frames

    out_dir = PROJECT_ROOT / 'mcu' / 'eval' / 'online' / args.board
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(out_dir / "timing_results.csv", index=False)


if __name__ == '__main__':
    main()