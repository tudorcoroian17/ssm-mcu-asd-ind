import hashlib
import json


def config_hash(cfg):
    canonical = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def train_config_hash(cfg, held_out_case):
    local_configs = {
        'held_out_case': held_out_case,
        'training': cfg['training'],
        'features': cfg['features'],
        'model': cfg['model'],
    }
    canonical = json.dumps(local_configs, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]