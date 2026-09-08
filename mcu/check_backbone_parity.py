"""
Compares a received on-device embedding against the golden pooled_mean
from parity_vectors_streaming.npz (05_phase_4_backbone_port.md step 4).

Only meaningful if the embedding came from the SAME clip the golden
reference was recorded from -- see PARITY_CLIP_STEM below, matching
runs/case1/16662b29beb3/parity_vectors_streaming_meta.json's "source_wav".
Prepare that clip with mcu/prepare_parity_clip.py, not prepare_clip.py's
random sample.

Usage:
    python mcu/check_backbone_parity.py
"""
from pathlib import Path

import numpy as np

RUN_DIR = Path('runs/case1/16662b29beb3')
EMBEDDING_DIR = Path('/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/audio_sender/clip_samples_logmel_parity/output')
PARITY_CLIP_STEM = '1200010003_ToyCar_case2_normal_IND_ch1_0003'


def main():
    on_device = np.load(EMBEDDING_DIR / f'{PARITY_CLIP_STEM}_embedding.npy')

    golden = np.load(RUN_DIR / 'parity_vectors_streaming.npz')
    reference = golden['pooled_mean'][0]  # (1, 64) -> (64,); one clip, not a batch

    abs_err = np.abs(on_device - reference)
    print(f'on-device : {on_device}')
    print(f'reference : {reference}')
    print(f'max abs error  : {abs_err.max():.6e}')
    print(f'mean abs error : {abs_err.mean():.6e}')
    print(f"(compare against findings/250's tolerance and the 0.000e+00 "
          f"streaming-vs-batched result in findings/220 section 9 -- this "
          f"number should be small, not exactly zero, since it now includes "
          f"real float32 differences between PyTorch and the C port, not "
          f"just a Python-side refactor)")


if __name__ == '__main__':
    main()