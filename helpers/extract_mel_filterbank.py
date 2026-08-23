import librosa
import numpy as np

from src.config import PROJECT_ROOT, load_config

_cfg = load_config()
_features = _cfg['features']

mel_fb = librosa.filters.mel(
    sr=_features['sample_rate'],
    n_fft=_features['n_fft'],
    n_mels=_features['n_mels'],
    fmin=_features['f_min'],
    fmax=_features['f_max'],
    htk=False,
    norm='slaney'
)
print(mel_fb.shape, mel_fb.dtype)
np.save(PROJECT_ROOT / 'configs' / 'mel_filterbank.npy', mel_fb)