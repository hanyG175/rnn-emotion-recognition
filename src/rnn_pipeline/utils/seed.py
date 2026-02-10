# Reproducibility functions
import os, random
import numpy as np
import torch
 
def set_seed(seed: int = 42) -> None:
    # Python built-in — affects random.shuffle, random.choice
    random.seed(seed)
 
    # NumPy — used internally by pandas transforms
    np.random.seed(seed)
 
    # PyTorch CPU operations
    torch.manual_seed(seed)
 
    # PyTorch GPU — all devices
    torch.cuda.manual_seed_all(seed)
 
    # Makes conv ops deterministic (slightly slower, fully reproducible)
    torch.backends.cudnn.deterministic = True
 
    # Disable auto-tuning; it picks different kernels between runs
    torch.backends.cudnn.benchmark = False
 
    # Worker seed — each DataLoader subprocess needs its own fixed seed
    os.environ["PYTHONHASHSEED"] = str(seed)
