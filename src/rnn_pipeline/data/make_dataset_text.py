import pandas as pd
from pathlib import Path
from data.preprocess_text import TextPreprocessor
from data.download import download_emotion_dataset

from utils.paths import PROCESSED_DIR, ARTIFACTS_DIR, RAW_DIR
from data.validation import validate_dataframe


def make_dataset(min_freq: int = 2
):
    """
    Load raw data, fit text preprocessor on training set,
    transform all splits, and save processed datasets + vocab.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure raw data exists
    required_files = ["train.parquet", "validation.parquet", "test.parquet"]
    if not all((RAW_DIR / f).exists() for f in required_files):
        print("Raw data not found. Downloading...")
        download_emotion_dataset()
    # ------------------
    # Load raw data
    # ------------------
    train_df = pd.read_parquet(RAW_DIR / "train.parquet")
    val_df = pd.read_parquet(RAW_DIR / "validation.parquet")
    test_df = pd.read_parquet(RAW_DIR / "test.parquet")

    validate_dataframe(train_df, "train")
    validate_dataframe(val_df, "val")
    validate_dataframe(test_df, "test")
    # ------------------
    # Preprocess
    # ------------------
    preprocessor = TextPreprocessor(min_freq=min_freq)

    preprocessed_train_df = preprocessor.fit(train_df).transform(train_df)
    preprocessor.save(ARTIFACTS_DIR / "vocab.json")

    preprocessed_val_df = preprocessor.transform(val_df)
    preprocessed_test_df = preprocessor.transform(test_df)

    # ------------------
    # Save processed data
    # ------------------
    preprocessed_train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    preprocessed_val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    preprocessed_test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print("Dataset creation complete.")
    
if __name__ == "__main__":

    make_dataset(min_freq=2
)
