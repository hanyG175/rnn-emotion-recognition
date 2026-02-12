# Config loading/validation (NEW)

import yaml
from pathlib import Path
 
REQUIRED_SECTIONS = {"data", "model", "training"}
REQUIRED_KEYS = {
    "data":     {"max_len"},
    "model":    {"embedding_dim", "hidden_dim", "dropout"},
    "training": {"batch_size", "num_epochs", "learning_rate"},
}
 
def load_config(config_path: Path | str) -> dict:
    path = Path(config_path)
 
    # Guard 1: file must exist
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
 
    with open(path) as f:
        config = yaml.safe_load(f)
 
    # Guard 2: required top-level sections
    missing = REQUIRED_SECTIONS - set(config.keys())
    if missing:
        raise KeyError(f"Missing config sections: {missing}")
 
    # Guard 3: required keys inside each section
    for section, keys in REQUIRED_KEYS.items():
        missing_keys = keys - set(config[section].keys())
        if missing_keys:
            raise KeyError(f"[{section}] missing: {missing_keys}")
 
    return config
