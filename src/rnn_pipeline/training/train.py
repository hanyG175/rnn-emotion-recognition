import json
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

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for text, label in train_loader:
        text, label = text.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

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



def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = pd.read_parquet("data/text/processed/train.parquet")
    val_dataset = pd.read_parquet("data/text/processed/val.parquet")
    
    # Load vocab
    with open("artifacts/vocab.json") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    num_classes = dataset["label"].nunique()

    # Load config
    with open("configs/rnn.yaml") as f:
        config = yaml.safe_load(f)

    # Datasets & loaders
    train_dataset = TextDataset(
        dataset,
        max_len=config["data"]["max_len"]
    )

    val_dataset = TextDataset(
        val_dataset,
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
    
    for epoch in range(config["training"]["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, num_classes, device)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - ", end="")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        improved = early_stopper.step(val_loss)

        if improved:
            torch.save(model.state_dict(), best_model_path)
            print("Validation improved — model saved")

        if early_stopper.should_stop:
            print("Early stopping triggered")
            break
            
    save_model(model, "artifacts/rnn_model.pt")


if __name__ == "__main__":
    main()
