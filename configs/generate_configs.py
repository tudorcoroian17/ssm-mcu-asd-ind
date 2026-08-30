import pandas as pd
import yaml

from runs.compute_hash import train_config_hash, config_hash
from src.config import load_config, PROJECT_ROOT

if __name__ == '__main__':
    base_dir = PROJECT_ROOT / 'configs' / 'ablation'
    base_dir.mkdir(parents=True, exist_ok=True)
    cases = [1, 2, 3, 4]
    d_states = [32, 16, 8]
    layers = [4, 2, 1]
    expand = [2, 1]
    selective = [True, False]
    discretization = ['zoh', 'euler']
    horizons = [2, 5]
    targets = ['residual', 'absolute']

    default_config = load_config()
    rows = []

    for d_state in d_states:
        for n_layers in layers:
            for exp in expand:
                for s in selective:
                    for d in discretization:
                        for h_k in horizons:
                            for target in targets:
                                default_config['model']['d_state'] = d_state
                                default_config['model']['n_layers'] = n_layers
                                default_config['model']['expand'] = exp
                                default_config['model']['selective'] = s
                                default_config['model']['discretization'] = d
                                default_config['training']['horizon_k'] = h_k
                                default_config['training']['target'] = target
                                config_name = f'{config_hash(default_config)}.yaml'
                                yaml.safe_dump(default_config, open(str(base_dir / config_name), 'w'))
                                for c in cases:
                                    config_hashed = train_config_hash(default_config, c)
                                    rows.append((c, d_state, n_layers, exp, s, d, h_k, target, config_name, config_hashed))

    df = pd.DataFrame(rows, columns=['held_out_case', 'd_state', 'n_layers', 'expand', 'selective', 'discretization', 'horizon_k', 'target', 'config_name', 'model_hash'])
    df.to_csv(base_dir / '000_config_manifest.csv', index=False)