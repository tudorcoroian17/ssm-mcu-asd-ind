"""
Compares a received on-device embedding against a golden mean-pooling
reference embedding for one deployment scheme
(06_phase5_nucleo_deployment_matrix.md).

Only meaningful if the embedding came from the SAME clip the golden
reference was recorded from -- see PARITY_CLIP_STEM below, matching
runs/case1/16662b29beb3/parity_vectors_streaming_meta.json's "source_wav".
Prepare that clip with mcu/prepare_clip.py.

The reference comes from one of two places, selected by --scheme-dir:
  - Omitted (default): the fp32 golden trace from
    runs/<case>/<hash>/parity_vectors_streaming.npz
    (05_phase_4_backbone_port.md step 4) -- use for the fp32 baseline.
  - --scheme-dir mcu/deploy/case<N>/<scheme>/: that scheme's
    reference_embedding.npy, produced by
    checks/smoke/weight_quant_parity.py --dump-reference-dir -- use for any
    quantized scheme. Comparing a quantized build against the fp32
    reference instead would misread the scheme's expected, validated
    embedding drift (findings/520-523) as a bug.

Usage:
    python mcu/check_backbone_parity.py
    python mcu/check_backbone_parity.py --scheme-dir mcu/deploy/case1/weight_int8_proj_pertensor
"""
import argparse
from pathlib import Path

import numpy as np

RUN_DIR = Path('runs/case1/16662b29beb3')
EMBEDDING_DIR = Path('/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/audio_sender/clip_samples_logmel_parity/output')
PARITY_CLIP_STEM = '1200010003_ToyCar_case2_normal_IND_ch1_0003'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme-dir', default=None,
                        help="mcu/deploy/case<N>/<scheme>/ directory holding "
                             "that scheme's reference_embedding.npy. Omit for "
                             "the fp32 baseline.")
    args = parser.parse_args()

    on_device = np.load(EMBEDDING_DIR / f'{PARITY_CLIP_STEM}_embedding.npy')

    if args.scheme_dir:
        reference = np.load(Path(args.scheme_dir) / 'reference_embedding.npy')
        label = args.scheme_dir
    else:
        golden = np.load(RUN_DIR / 'parity_vectors_streaming.npz')
        reference = golden['pooled_mean'][0]  # (1, 64) -> (64,); one clip, not a batch
        label = 'fp32 baseline'

    abs_err = np.abs(on_device - reference)
    print(f'scheme    : {label}')
    print(f'on-device : {on_device}')
    print(f'reference : {reference}')
    print(f'max abs error  : {abs_err.max():.6e}')
    print(f'mean abs error : {abs_err.mean():.6e}')


if __name__ == '__main__':
    main()