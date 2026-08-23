import librosa
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

_cfg = load_config()
_features = _cfg['features']
SR = _features['sample_rate']
N_FFT = _features["n_fft"]
HOP = _features["hop_length"]
LOG_EPS = _features["log_eps"]
CENTER = _features["center"]

mel_fb = np.load(PROJECT_ROOT / "configs" / "mel_filterbank.npy")

def extract_logmel(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)
    stft = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP, window='hann', center=True)
    power_spec = np.abs(stft) ** 2
    mel_spec = mel_fb @ power_spec
    logmel = np.log(mel_spec + LOG_EPS)
    return logmel.T

def compute_raw_frame_rms(wav_path):
    """
    Per-frame RMS of the RAW waveform, framed to line up exactly with
    how extract_logmel's STFT frames it (center=True pads by N_FFT//2
    on each side before framing) -- so frame i here corresponds to
    frame i of the cached log-mel array, same T_total for the same file.

    Pad mode doesn't need to match librosa's internal choice exactly --
    unlike the SSM scan's parity requirement, we only need frame COUNT
    to align (for indexing) and approximate per-frame energy (for
    silence detection), not bit-exact values at the padded edges.
    """
    y, _ = librosa.load(wav_path, sr=SR)
    y_padded = np.pad(y, N_FFT // 2, mode='constant')
    frames = librosa.util.frame(y_padded, frame_length=N_FFT, hop_length=HOP).T
    return np.sqrt((frames ** 2).mean(axis=-1))

if __name__ == '__main__':
    MANIFEST_PATH = PROJECT_ROOT / 'manifest.csv'

    manifest = pd.read_csv(MANIFEST_PATH)
    sample_path = manifest[manifest['source'] == 'IND'].iloc[0]['path']
    log_mel = extract_logmel(sample_path)
    print("shape:", log_mel.shape, "(expect (344, 64))")
    print("min/max:", log_mel.min(), log_mel.max())