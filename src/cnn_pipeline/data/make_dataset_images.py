import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/images/raw/legend.csv")
root_dir = "data/images/raw/images/"

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"])

train_df.to_parquet("data/images/processed/train.parquet")
val_df.to_parquet("data/images/processed/val.parquet")
test_df.to_parquet("data/images/processed/test.parquet")