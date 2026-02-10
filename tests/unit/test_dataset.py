# tests/unit/test_dataset.py
from src.rnn_pipeline.data.datasets import TextDataset


def test_padding_to_max_len(sample_df):
    # Short sequence (2 tokens) must be zero-padded to max_len=10
    ds = TextDataset(sample_df, max_len=10)
    tokens, _ = ds[1]
    assert tokens.shape[0] == 10
 
def test_truncation_to_max_len(sample_df):
    # Long sequence (7 tokens) must be truncated to max_len=4
    ds = TextDataset(sample_df, max_len=4)
    tokens, _ = ds[2]
    assert tokens.shape[0] == 4
