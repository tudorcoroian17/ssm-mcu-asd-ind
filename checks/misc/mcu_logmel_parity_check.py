"""
Compares MCU log-mel output against the Python reference and characterizes
where the disagreement lives.

Usage:
    python -m checks.misc.mcu_logmel_parity_check <wav_path> <mcu_output_npy_path>
"""
import sys
import os
from pathlib import Path

import numpy as np

from src.features.logmel import extract_logmel

TOLERANCE = 0.2627  # findings/250


def main(wav_path, mcu_npy_path):
    ref_dir = mcu_npy_path.parents[1] / 'reference'
    mcu = np.load(mcu_npy_path)
    ref = extract_logmel(wav_path)

    print(f'MCU shape: {mcu.shape}, reference shape: {ref.shape}')
    if mcu.shape != ref.shape:
        print('FAIL: shape mismatch')
        return

    # Save the reference beside the MCU output for offline inspection.
    ref_path = ref_dir / mcu_npy_path.name
    np.save(ref_path, ref.astype(np.float32))
    print(f'Reference saved to {ref_path}')

    diff = mcu - ref
    absdiff = np.abs(diff)
    bad = absdiff > TOLERANCE

    print(f'\nMax abs diff: {absdiff.max():.6f}  (tolerance {TOLERANCE})')
    print(f'Failing points: {bad.sum()} of {bad.size} ({bad.mean():.4%})')
    print(f'Mean abs diff: {absdiff.mean():.6f}')
    print(f'Signed mean diff: {diff.mean():+.6f}  '
          f'(near zero = symmetric noise; large = systematic offset)')

    print('\nFailures per bin (bins with any failure):')
    per_bin = bad.sum(axis=0)
    for b in np.nonzero(per_bin)[0]:
        print(f'  bin {b:2d}: {per_bin[b]:4d} failures, '
              f'max diff {absdiff[:, b].max():.4f}, '
              f'mean signed {diff[bad[:, b], b].mean():+.4f}')

    print('\nFailures per frame (first 30 frames with any failure):')
    per_frame = bad.sum(axis=1)
    bad_frames = np.nonzero(per_frame)[0]
    print(f'  total frames with failures: {len(bad_frames)} of {mcu.shape[0]}')
    for f in bad_frames[:30]:
        print(f'  frame {f:3d}: {per_frame[f]:2d} failing bins, '
              f'max diff {absdiff[f].max():.4f}')

    if len(bad_frames) > 1:
        gaps = np.diff(bad_frames)
        vals, counts = np.unique(gaps, return_counts=True)
        print('\nGaps between consecutive failing frames (spacing -> count):')
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:8]:
            print(f'  {v} -> {c}')

    print('\nWorst 10 points:')
    flat = np.argsort(absdiff, axis=None)[::-1][:10]
    for idx in flat:
        f, b = np.unravel_index(idx, absdiff.shape)
        print(f'  frame {f:3d} bin {b:2d}: mcu={mcu[f, b]:9.4f} '
              f'ref={ref[f, b]:9.4f} diff={diff[f, b]:+8.4f}')

    print('\nPASS' if absdiff.max() <= TOLERANCE else '\nFAIL')


if __name__ == '__main__':
    dir_npy = Path('/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/audio_sender/clip_samples_logmel_parity')
    original_dir = dir_npy / 'original'
    output_dir = dir_npy / 'output'
    npy_files = output_dir.glob('*.npy')
    for file in npy_files:
        main(original_dir / f'{file.stem}.wav', file)