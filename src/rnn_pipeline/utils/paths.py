from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "text/raw"
PROCESSED_DIR = DATA_DIR / "text/processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
CONFIG_DIR = PROJECT_ROOT / "configs"

