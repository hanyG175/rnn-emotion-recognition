# Trainer class (NEW)
import json
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from data.datasets import TextDataset
from models.rnn import TextClassifier
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping
from torchmetrics.classification import Accuracy

from tqdm import tqdm
from datetime import datetime
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.metric import MetricsLogger
from utils.monitoring import ExperimentTracker
from utils.seed import set_seed
from schedulers import get_scheduler, scheduler_step
from utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, VOCAB_PATH, CONFIG_DIR

logger = get_logger(__name__) 

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
        # clip gradient norm to 1.0; prevents RNN exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
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
    # Setting up reproducibility, device, and loading data
    config = load_config(CONFIG_DIR / "rnn.yaml")      
    set_seed(config["training"].get("seed", 42))    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    
    # Load vocab
    with open(VOCAB_PATH) as f:
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
    # -----------------------------------
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

    best_model_path = ARTIFACTS_DIR / "best_model.pt"

    #MLflow/W&B integration
    run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}"          
    tracker  = ExperimentTracker("rnn-text-classification")  
    tracker.start_run(run_name=run_name)                           
    tracker.log_params(config)                                         
    
    # Set generator for reproducible 'shuffling' in DataLoader because shuffle=True uses a random seed internally. This ensures the same shuffling order across runs with the same seed.
    g = torch.Generator(); 
    g.manual_seed(config["training"]["seed"])
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, generator=g)
 
    scheduler    = get_scheduler(optimizer, config) # create LR scheduler based on config              
    metrics_log  = MetricsLogger(f"{ARTIFACTS_DIR}/metrics/{run_name}.csv") # log metrics to CSV for later analysis
    
    try: 
        for epoch in range(config["training"]["num_epochs"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, num_classes, device)
            current_lr = optimizer.param_groups[0]["lr"]        
            logger.info(f"Epoch {epoch+1} | loss={train_loss:.4f} |val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={current_lr:.2e}")
            metrics_log.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}) 
            tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            scheduler_step(scheduler, val_loss)                       
            if early_stopper.step(val_loss):
                save_checkpoint(model, optimizer, epoch,
                    val_loss, val_acc, config, str(best_model_path))
            if early_stopper.should_stop:
                break
    finally:
        tracker.end_run()                                             
        metrics_log.close()                                           
        
        
if __name__ == "__main__":
    main()
