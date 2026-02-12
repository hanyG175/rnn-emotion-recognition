 # Custom metrics
import torch
from torchmetrics.classification import Accuracy, F1Score

def calculate_metrics(model, dataloader, device="cpu"):
    model.to(device)
    acc = Accuracy(task="multiclass", num_classes=dataloader.dataset.num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=dataloader.dataset.num_classes).to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for text, labels in dataloader:
            text, labels = text.to(device), labels.to(device)
            preds = model(text).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            acc.update(preds, labels)
            f1.update(preds, labels)

    metrics = {
        "accuracy": acc.compute().item(),
        "f1": f1.compute().item(),
    }
    return metrics
