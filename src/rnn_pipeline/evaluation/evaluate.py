import torch
import pandas as pd
import json
from torch.utils.data import DataLoader
from ..data.datasets import TextDataset
from ..models.rnn import TextClassifier
from .metrics import calculate_metrics

def evaluate_model(model_path, data_path, vocab_path, config):
    # 1. Load data
    df = pd.read_parquet(data_path)
    
    # 2. Load vocab
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    # 3. Dataset & DataLoader
    dataset = TextDataset(df, max_len=config["data"]["max_len"])
    loader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False)
    
    # 4. Load model
    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=df["label"].nunique(),
        dropout_prob=config["model"]["dropout"]
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 5. Inference & metrics
    metrics = calculate_metrics(model, loader)
    
    print(metrics)
    return metrics
