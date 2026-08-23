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

if __name__ == '__main__':
    MANIFEST_PATH = PROJECT_ROOT / 'manifest.csv'

    manifest = pd.read_csv(MANIFEST_PATH)
    sample_path = manifest[manifest['source'] == 'IND'].iloc[0]['path']
    log_mel = extract_logmel(sample_path)
    print("shape:", log_mel.shape, "(expect (344, 64))")
    print("min/max:", log_mel.min(), log_mel.max())