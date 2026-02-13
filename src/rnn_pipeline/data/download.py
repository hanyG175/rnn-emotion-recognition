from utils.paths import RAW_DIR
import pandas as pd


SPLITS = {
    "train": "split/train-00000-of-00001.parquet",
    "validation": "split/validation-00000-of-00001.parquet",
    "test": "split/test-00000-of-00001.parquet",
}

BASE_URL = "hf://datasets/dair-ai/emotion/"

def download_emotion_dataset():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for split, path in SPLITS.items():
        save_path = RAW_DIR / f"{split}.parquet"

        if save_path.exists():
            print(f"{split} already exists. Skipping.")
            continue

        print(f"Downloading {split}...")
        df = pd.read_parquet(BASE_URL + path)
        df.to_parquet(save_path, index=False)

    print("Download complete.")