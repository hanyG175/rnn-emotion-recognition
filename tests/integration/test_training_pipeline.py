# tests/integration/test_training_pipeline.py

from src.rnn_pipeline.data.datasets import TextDataset
from src.rnn_pipeline.models.rnn import TextClassifier
from src.rnn_pipeline.training.trainer import train_one_epoch, validate
from src.rnn_pipeline.training.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_full_pipeline_runs(tiny_config, tiny_data):
    # Build all components from scratch with synthetic data
    dataset = TextDataset(tiny_data, max_len=10)
    loader  = DataLoader(dataset, batch_size=4)
    model   = TextClassifier(vocab_size=30, embed_dim=16, hidden_dim=16, num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    stopper = EarlyStopping(patience=2, min_delta=0.0)
 
    for _ in range(2):
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, loader, criterion, 3, device)
        stopper.step(val_loss)
 
    # Not testing accuracy — testing that nothing crashes
    assert 0.0 < train_loss < 10.0 # type: ignore
    assert 0.0 <= val_acc <= 1.0 # pyright: ignore[reportPossiblyUnboundVariable]