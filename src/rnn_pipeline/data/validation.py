import pandas as pd

from src.rnn_pipeline.utils import logger
 
REQUIRED_COLUMNS  = {"text", "label"}
MIN_SAMPLES_CLASS = 50
MAX_IMBALANCE     = 10.0
 
def validate_dataframe(df: pd.DataFrame, split_name: str = "dataset"):
 
    # 1. Not empty
    if df.empty:
        raise ValueError(f"[{split_name}] Dataset is empty.")
 
    # 2. Required columns present
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"[{split_name}] Missing columns: {missing}")
 
    # 3. No nulls — NaN deep in a training loop gives cryptic errors
    nulls = df[list(REQUIRED_COLUMNS)].isnull().sum()
    if nulls.any():
        raise ValueError(f"[{split_name}] Null values: {nulls[nulls > 0]}")
 
    # 4. Class imbalance — warn, do not raise
    counts = df["label"].value_counts()
    ratio  = counts.max() / max(counts.min(), 1)
    if ratio > MAX_IMBALANCE:
        logger.warning(f"[{split_name}] Imbalance ratio = {ratio:.1f}x")
