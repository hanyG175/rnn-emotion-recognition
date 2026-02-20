import torch
from torchmetrics.classification import Accuracy, F1Score


def calculate_metrics(model, dataloader, num_classes, device="cpu"):
    model.to(device)
    model.eval()

    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, labels in dataloader:
            text = text.to(device)
            labels = labels.to(device)

            logits = model(text)
            preds = torch.argmax(logits, dim=1)

            acc.update(preds, labels)
            f1.update(preds, labels)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # 🔑 concatenate into flat tensors
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = {
        "accuracy": acc.compute().item(),
        "f1": f1.compute().item(),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }

    return metrics