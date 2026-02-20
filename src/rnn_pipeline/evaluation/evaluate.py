import json, sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
# from evaluation.metrics import calculate_metrics
from data.datasets import TextDataset
from data.validation import validate_dataframe
from models.rnn import TextClassifier
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, CONFIG_DIR

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

    # 5. Calculate metrics
    metrics = calculate_metrics(model, loader)
    
    print(metrics)
    return metrics


 
logger = get_logger(__name__)
 
def main():
    # Parse command line: python -m ... evaluate.py <checkpoint_path>
    if len(sys.argv) > 2:
        print("Usage: python -m rnn_pipeline.evaluation.evaluate <checkpoint_path>")
        sys.exit(1)
 
    checkpoint_path = ARTIFACTS_DIR / "best_model.pt"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
 
    logger.info(f"Evaluating: {checkpoint_path}")
 
    # Load config, test data, vocab
    config = load_config( CONFIG_DIR / "rnn.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
 
    with open(ARTIFACTS_DIR / "vocab.json") as f:
        vocab = json.load(f)
 
    vocab_size  = len(vocab)
    num_classes = test_df["label"].nunique()
 
    # Build test loader
    test_dataset = TextDataset(test_df, max_len=config["data"]["max_len"])
    test_loader  = DataLoader(test_dataset, batch_size=config["training"]["batch_size"],
                                   shuffle=False, num_workers=2)
 
    # Instantiate model and load checkpoint
    model = TextClassifier(vocab_size, config["model"]["embedding_dim"],
                         config["model"]["hidden_dim"], num_classes,
                         config["model"]["dropout"]).to(device)
 
    ckpt = load_checkpoint(checkpoint_path, model, device=device)
    logger.info(f"Loaded epoch {ckpt['epoch']}")
 
    # Run evaluation
    results = evaluate_model(model, test_loader, num_classes, device)
 
    # Print metrics
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"F1 Score:  {results['f1']:.4f}")
    logger.info("=" * 60)
 
    # Generate classification report
    report = classification_report(results["all_labels"], results["all_preds"], digits=4)
    print(report)
 
    # Save confusion matrix + report
    cm_path = Path(ARTIFACTS_DIR) / Path("confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(results["confusion_matrix"]))
        f.write("\n\nClassification Report:\n")
        f.write(report)
 
    logger.info(f"Results saved → {cm_path}")
 
 
if __name__ == "__main__":
    main()
