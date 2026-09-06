import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

configs_root = PROJECT_ROOT / 'configs' / 'ablation'
runs_root = PROJECT_ROOT / 'runs'
configs_manifest = pd.read_csv(configs_root / '000_config_manifest.csv')
configs_for_check = configs_manifest[
        (configs_manifest['config_name'].isin(['f4cd557b7e3b.yaml', 'f2578cb06991.yaml',
                                              '352f70960ed3.yaml', 'b39731b66741.yaml']))
    ]

for index, row in configs_for_check.iterrows():
    case = int(row['held_out_case'])
    model_hash = row['model_hash']
    ckpt_path = runs_root / f'case{case}' / model_hash / 'ckpt.pt'

    if not ckpt_path.exists():
        print(f'model {model_hash} not cached')
        continue

    parity_parallel = runs_root / f'case{case}' / model_hash / 'parity_vectors.npz'
    parity_streaming = runs_root / f'case{case}' / model_hash / 'parity_vectors_streaming.npz'

    if not parity_streaming.exists():
        print(f'parity_streaming not cached for case {case} model hash {model_hash}')
        continue

    if not parity_parallel.exists():
        print(f'parity_parallel not cached for case {case} model hash {model_hash}')
        continue

    print(f'running for case {case} model hash {model_hash}')

    old = np.load(parity_parallel)
    new = np.load(parity_streaming)

    assert set(old.files) == set(new.files), set(old.files) ^ set(new.files)
    for key in sorted(old.files):
        diff = np.abs(old[key] - new[key]).max()
        flag = '  <-- CHECK' if diff > 1e-5 else ''
        print(f'{key:30s} {diff:.3e}{flag}')