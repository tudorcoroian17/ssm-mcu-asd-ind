"""
Exports the frozen feature-pipeline contract (Phase 1 output) to a C header
for the MCU feature pipeline (04_phase_3_mcu_feature_pipeline.md §3.1).

Shared across all three boards -- the filterbank and window are the same
regardless of target; per-board differences (DSP library, log implementation)
live in the §3.2/§3.3 code, not here.

Run once, whenever configs/mel_filterbank.npy or configs/default.yaml change.
"""
import numpy as np
from src.config import PROJECT_ROOT, load_config

cfg = load_config()
feat = cfg['features']

mel_fb = np.load(PROJECT_ROOT / 'configs' / 'mel_filterbank.npy')  # (n_mels, n_freq_bins)
n_mels, n_freq = mel_fb.shape
n_fft = feat['n_fft']
assert n_freq == n_fft // 2 + 1


def sparse_runs(fb):
    """One contiguous (start, length, values) run per mel bin.

    Verified on this project's actual filterbank (findings/260): every row's
    nonzero entries form one contiguous run. GAP 1: if mel_filterbank.npy is
    ever regenerated with different parameters, keep the assertion below --
    a non-triangular filter shape would otherwise fail silently and produce
    wrong C code.
    """
    starts, lengths, values = [], [], []
    for row in fb:
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            starts.append(0); lengths.append(0)
            continue
        start, end = nz.min(), nz.max()
        assert (end - start + 1) == len(nz), "non-contiguous filter row"
        starts.append(int(start))
        lengths.append(int(end - start + 1))
        values.extend(row[start:end + 1].tolist())
    return starts, lengths, values


starts, lengths, values = sparse_runs(mel_fb)
offsets = np.concatenate([[0], np.cumsum(lengths)]).tolist()


def periodic_hann(n):
    # Matches librosa.stft(window='hann'): periodic/DFT-even Hann, not the
    # symmetric variant. N in the denominator, not N-1.
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).tolist()


window = periodic_hann(n_fft)

# GAP 2: everything below is float32. If the RP2040's real-time margin (§3.6)
# comes up short, this table is the place to revisit q15 fixed-point --
# don't pre-optimize before you've actually measured a shortfall.


def emit_c_array(name, values, ctype='float'):
    if ctype == 'float':
        body = ', '.join(f'{v:.9g}f' for v in values)
    else:
        body = ', '.join(str(v) for v in values)
    return f'static const {ctype} {name}[{len(values)}] = {{ {body} }};\n'


with open('mcu_feature_contract.h', 'w') as f:
    f.write('#pragma once\n\n')
    f.write(f'#define MEL_N_MELS {n_mels}\n')
    f.write(f'#define MEL_N_FREQ_BINS {n_freq}\n')
    f.write(f'#define MEL_N_FFT {n_fft}\n')
    f.write(f'#define MEL_HOP_LENGTH {feat["hop_length"]}\n')
    f.write(f'#define MEL_LOG_EPS {feat["log_eps"]}f\n\n')
    f.write(emit_c_array('mel_starts', starts, 'unsigned short'))
    f.write(emit_c_array('mel_lengths', lengths, 'unsigned short'))
    f.write(emit_c_array('mel_offsets', offsets, 'unsigned short'))
    f.write(emit_c_array('mel_values', values, 'float'))
    f.write(emit_c_array('hann_window', window, 'float'))

print(f'wrote mcu_feature_contract.h: {len(values)} filterbank values, '
      f'{n_fft} window samples')