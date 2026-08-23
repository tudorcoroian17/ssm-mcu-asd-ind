import hashlib
import json
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from tqdm import tqdm

from src.config import load_config, PROJECT_ROOT
from src.features.logmel import extract_logmel

MANIFEST_PATH = PROJECT_ROOT / 'manifest.csv'
CACHE_ROOT = PROJECT_ROOT / 'cache'

def config_hash(feature_cfg):
    canonical = json.dumps(feature_cfg, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def build_cache():
    _cfg = load_config()
    _features = _cfg['features']
    h = config_hash(_features)
    cache_dir = CACHE_ROOT / h
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(CACHE_ROOT / f'{h}_feature_config.yaml', 'w') as f:
        yaml.safe_dump(_features, f)

    manifest = pd.read_csv(MANIFEST_PATH)
    cache_paths = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        wav_path = Path(row['path'])
        cache_path = cache_dir / f'{wav_path.stem}.npy'

        if not cache_path.exists():
            log_mel = extract_logmel(str(wav_path))
            np.save(cache_path, log_mel.astype(np.float32))
        cache_paths.append(cache_path)

    manifest["cache_path"] = cache_paths
    manifest["feature_config_hash"] = h
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"cached {len(manifest)} files under {cache_dir}")

if __name__ == "__main__":
    build_cache()