import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return accuracy_score(y_true, y_pred)
