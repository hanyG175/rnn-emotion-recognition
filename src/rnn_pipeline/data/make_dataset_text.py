import pandas as pd
from pathlib import Path
from preprocess_text import TextPreprocessor

from utils.paths import PROCESSED_DIR, ARTIFACTS_DIR, RAW_DIR
from .validation import validate_dataframe


def make_dataset(
    train_path: Path | str,
    val_path: Path | str,
    test_path: Path | str,
    processed_dir: Path,
    artifacts_dir: Path,
    min_freq: int = 2
):
    """
    Load raw data, fit text preprocessor on training set,
    transform all splits, and save processed datasets + vocab.
    """

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------
    # Load raw data
    # ------------------
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    validate_dataframe(train_df, "train")
    validate_dataframe(val_df, "val")
    validate_dataframe(test_df, "test")
    # ------------------
    # Preprocess
    # ------------------
    preprocessor = TextPreprocessor(min_freq=min_freq)

    preprocessed_train_df = preprocessor.fit(train_df).transform(train_df)
    preprocessor.save(artifacts_dir / "vocab.json")

    preprocessed_val_df = preprocessor.transform(val_df)
    preprocessed_test_df = preprocessor.transform(test_df)

    # ------------------
    # Save processed data
    # ------------------
    preprocessed_train_df.to_parquet(processed_dir / "train.parquet", index=False)
    preprocessed_val_df.to_parquet(processed_dir / "val.parquet", index=False)
    preprocessed_test_df.to_parquet(processed_dir / "test.parquet", index=False)

    print("Dataset creation complete.")
    
if __name__ == "__main__":
    make_dataset(
        train_path=RAW_DIR / "train-00000-of-00001.parquet",
        val_path=RAW_DIR / "validation-00000-of-00001.parquet",
        test_path=RAW_DIR / "test-00000-of-00001.parquet",
        processed_dir=PROCESSED_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        min_freq=2
    )
