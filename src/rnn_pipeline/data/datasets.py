import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class TextDataset(Dataset):
  def __init__(self, dataframe, max_len=100):
    self.data = dataframe
    self.max_len = max_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    seq = self.data.iloc[idx]["numericalized"]
    label = self.data.iloc[idx]["label"]

    # Pad or truncate sequence to fixed length
    if len(seq) < self.max_len:
      seq = np.pad(seq, (0, self.max_len - len(seq)), 'constant', constant_values=0) # pad with <PAD>=0
    else:
      seq = seq[:self.max_len]

    return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)
  
