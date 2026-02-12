import pandas as pd
import sys


from utils.logger import get_logger
logger = get_logger(__name__)


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
        logger.warning(f"[{split_name}] Imbalance ratio = {ratio:.1f}x") # type: ignore

    # 5. Too few samples in any class — warn, do not raise
    if (counts < MIN_SAMPLES_CLASS).any():
        small = counts[counts < MIN_SAMPLES_CLASS]
        logger.warning(f"[{split_name}] Classes with <{MIN_SAMPLES_CLASS} samples: {small.to_dict()}")

    # 6. Text column is string
    if not pd.api.types.is_string_dtype(df["text"]):
        raise TypeError(f"[{split_name}] 'text' column must be string.")

    # 7. Label column is categorical/string — if numeric, we assume it's already encoded
    if df["label"].dtype == object:
        logger.info(f"[{split_name}] Label column is categorical/string.")

    # 8. No duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"[{split_name}] Found {duplicates} duplicate rows.")
