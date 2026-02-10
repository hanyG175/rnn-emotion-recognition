 # LR scheduling (NEW)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
 
def get_scheduler(optimizer, config: dict):
    sched_cfg = config["training"].get("scheduler")
    if sched_cfg is None: return None
 
    sched_type = sched_cfg.get("type", "").lower()
 
    if sched_type == "reduce_on_plateau":
        # Reduce LR after N epochs of no val_loss improvement
        return ReduceLROnPlateau(
            optimizer, mode="min",
            patience=sched_cfg.get("patience", 2),
            factor=sched_cfg.get("factor", 0.5),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
 
    elif sched_type == "cosine":
        # Smooth cosine decay over all epochs
        return CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
 
def scheduler_step(scheduler, val_loss: float) -> None:
    if scheduler is None: return
    # ReduceLROnPlateau needs the metric; other schedulers just step()
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
