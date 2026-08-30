from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(PROJECT_ROOT / 'configs' / 'default.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_config_by_name(name):
    with open(PROJECT_ROOT / 'configs' / 'ablation' / name, 'r') as f:
        return yaml.safe_load(f)