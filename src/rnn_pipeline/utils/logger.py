# Custom logging setup
import logging, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
 
def get_logger(
    name: str,
    log_dir: str = "artifacts/logs",
    level: int = logging.INFO,
) -> logging.Logger:
 
    # Returns same object if called twice — prevents duplicate handlers
    logger = logging.getLogger(name)
    if logger.handlers: return logger
 
    logger.setLevel(level)
 
    # Format: 2026-02-10 14:30:00 | train | INFO | Training started
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
 
    # Console handler — stdout is captured by Docker / cloud agents
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
 
    # Rotating file: 5 MB max, keeps 3 backup files
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        Path(log_dir) / "training.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
