# Trainer class (NEW)
import json
from jsonschema import validate
import torch
import yaml
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.rnn_pipeline.training.early_stopping import EarlyStopping
from torchmetrics import Accuracy

from ..data.datasets import TextDataset
from ..models.rnn import TextClassifier

# NEW imports — add these to the top of train.py
from tqdm import tqdm
from datetime import datetime
from rnn_pipeline.data.validation import validate_dataframe
from rnn_pipeline.utils.checkpoint import save_checkpoint
from rnn_pipeline.utils.config import load_config
from rnn_pipeline.utils.logger import get_logger
from rnn_pipeline.utils.metric import MetricsLogger
from rnn_pipeline.utils.monitoring import ExperimentTracker
from rnn_pipeline.utils.seed import set_seed
from rnn_pipeline.training.schedulers import get_scheduler, scheduler_step
 
logger = get_logger(__name__)   # NEW — replaces print()

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc="  Train", leave=False)  # NEW
    for text, label in loop:
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        # NEW — clip gradient norm to 1.0; prevents RNN exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")  # NEW
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, num_classes, device):
    model.eval()

    val_loss = 0.0
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for text, label in val_loader:
            text, label = text.to(device), label.to(device)

            output = model(text)
            loss = criterion(output, label)

            val_loss += loss.item()
            acc.update(output.argmax(dim=1), label)

    return val_loss / len(val_loader), acc.compute().item()


def main():
    config = load_config("configs/rnn.yaml")          # NEW
    set_seed(config["training"].get("seed", 42))     # NEW
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df   = pd.read_parquet("data/processed/val.parquet")
    validate_dataframe(train_df, "train")                       # NEW
    validate_dataframe(val_df,   "val")   
    
    
    # Load vocab
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)
        
    vocab_size = len(vocab)
    num_classes = train_df["label"].nunique()
    
    # Datasets & loaders
    train_dataset = TextDataset(
        train_df,
        max_len=config["data"]["max_len"]
    )

    val_dataset = TextDataset(
        val_df,
        max_len=config["data"]["max_len"]
    )
    # -------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # Model
    model = TextClassifier(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=num_classes,
        dropout_prob=config["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    # Train
    early_stopper = EarlyStopping(
    patience=config["training"]["early_stopping"]["patience"],
    min_delta=config["training"]["early_stopping"]["min_delta"]
)

    best_model_path = "artifacts/best_model.pt"

    run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}"           # NEW
    tracker  = ExperimentTracker("rnn-text-classification")   # NEW
    tracker.start_run(run_name=run_name)                            # NEW
    tracker.log_params(config)                                    # NEW
 
                      # NEW
 
    # NEW — seeded generator for deterministic DataLoader shuffle
    g = torch.Generator(); g.manual_seed(config["training"]["seed"])
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, generator=g)
 
    scheduler    = get_scheduler(optimizer, config)                 # NEW
    metrics_log  = MetricsLogger(f"artifacts/metrics/{run_name}.csv") # NEW
 
    for epoch in range(config["training"]["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, num_classes, device)
        current_lr = optimizer.param_groups[0]["lr"]         # NEW
        logger.info(f"Epoch {epoch+1} | loss={train_loss:.4f} | lr={current_lr:.2e}")
        metrics_log.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})  # NEW
        tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch) # NEW
        scheduler_step(scheduler, val_loss)                        # NEW
        if early_stopper.step(val_loss):
            save_checkpoint(model, optimizer, epoch,
                val_loss, val_acc, config, str(best_model_path)) # NEW
 
    tracker.end_run()                                              # NEW
    metrics_log.close()                                            # NEW
