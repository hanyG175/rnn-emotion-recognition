# Save/load checkpoints (NEW)
import torch
from datetime import datetime
from pathlib import Path
 
def save_checkpoint(model, optimizer, epoch, val_loss,
                      val_acc, config, path: str) -> None:
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        # optimizer_state_dict stores momentum buffers — required to resume
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
        "val_acc":              val_acc,
        # embed config in checkpoint — file is self-documenting
        "config":               config,
        "saved_at":             datetime.now().isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
 
def load_checkpoint(path, model, optimizer=None,
                      device=None) -> dict:
    # map_location: load GPU checkpoint on CPU machine without error
    map_loc = device if device else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_loc, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
