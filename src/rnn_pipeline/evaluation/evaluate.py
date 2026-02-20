import json
import sys
from pathlib import Path

import pandas as pd
# import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from .confusion_matrix import plot_confusion_matrix
from ..data.datasets import TextDataset
from ..models.rnn import TextClassifier
from .metrics import calculate_metrics
from ..utils.checkpoint import load_checkpoint
from ..utils.config import load_config
from ..utils.logger import get_logger
from ..utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, CONFIG_DIR

logger = get_logger(__name__)


def evaluate_model(model, dataloader, num_classes, device):
    model.eval()
    with torch.no_grad():
        metrics = calculate_metrics(model, dataloader, num_classes, device)
    return metrics


def main():
    # Usage check
    if len(sys.argv) > 2:
        print("Usage: python -m rnn_pipeline.evaluation.evaluate <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = ARTIFACTS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Evaluating: {checkpoint_path}")

    # Load config
    config = load_config(CONFIG_DIR / "rnn.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    # Load vocab
    with open(ARTIFACTS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    num_classes = test_df["label"].nunique()

    # Dataset & DataLoader
    test_dataset = TextDataset(
        test_df,
        max_len=config["data"]["max_len"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2
    )

    # Build model
    model = TextClassifier(
        vocab_size=vocab_size,
        embed_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=num_classes,
        dropout_prob=config["model"]["dropout"]
    ).to(device)

    # Load checkpoint
    ckpt = load_checkpoint(checkpoint_path, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Run evaluation
    results = evaluate_model(model, test_loader,num_classes, device)

    # Log metrics
    logger.info("=" * 60)
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")
    logger.info("=" * 60)

    # Classification report
    report = classification_report(
        results["all_labels"],
        results["all_preds"],
        zero_division=0,
        digits=4
    )
    print(report)
   

    labels = list(range(num_classes))

    cm_image_path = ARTIFACTS_DIR / "confusion_matrix.png"
    cm = plot_confusion_matrix(
        results["all_labels"],
        results["all_preds"],
        labels=labels,
        save_path=cm_image_path
    )
    
    # Save results
    cm_path = ARTIFACTS_DIR / "confusion_matrix.txt"
    with open(cm_path, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write("\n\nClassification Report:\n")
        f.write(report)

    logger.info(f"Results saved → {cm_path}")


if __name__ == "__main__":
    main()