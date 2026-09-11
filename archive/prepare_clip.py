"""
-----------------------------------------------------------------------------
**ARCHIVED** - Functionality of resampling at 16MHz is baked into the test
harness script now. This version is no longer maintained or supported.
-----------------------------------------------------------------------------
Prepares a clip for the MCU parity test.

Loads the WAV with exactly the same call src/features/logmel.py uses, so the
samples streamed to the board are the samples the reference log-mel is computed
from. Otherwise a difference in loading or resampling appears as a
feature-pipeline mismatch that isn't one.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import shutil

from src.features.logmel import SR


def main(wav_path):
    out_path = Path('/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/audio_sender/clip_samples_logmel_parity/input') / f'{Path(wav_path).stem}.npy'
    info = sf.info(wav_path)
    print(f'Native: {info.samplerate} Hz, {info.channels} ch, '
          f'{info.subtype}, {info.frames} frames')

    y, _ = librosa.load(wav_path, sr=SR)
    print(f'After librosa.load(sr={SR}): {len(y)} samples, peak {np.abs(y).max():.6f}')
    if info.samplerate != SR:
        print(f'NOTE: resampled {info.samplerate} -> {SR}')

    # Round, don't truncate: .astype() truncates toward zero, which biases
    # every sample by up to 1 LSB.
    y_int16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype('<i2')
    np.save(out_path, y_int16)
    print(f'Saved {len(y_int16)} int16 samples to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_name', type=str, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv('manifest.csv')
    ind_clips = manifest[manifest['source'] == 'IND']
    clips_input = [clip['path'] for _, clip in ind_clips.sample(5).iterrows()]

    if args.clip_name:
        clips_input = [Path('/mnt/d/Tudor/Master/Disertatie/ToyCar-ToyADMOS-DS/case2/NormalSound_IND') / args.clip_name]

    for clip in clips_input:
        print(f'Processing {clip}')
        original = Path('/mnt/d/Tudor/Master/Disertatie/ssm-mcu-asd-deploy/audio_sender/clip_samples_logmel_parity/original') / Path(clip).name
        shutil.copy(clip, original)
        main(clip)