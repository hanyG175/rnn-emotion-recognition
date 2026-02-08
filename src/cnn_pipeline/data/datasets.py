import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset


class FacialExpressionDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.data = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(self.data['emotion'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        image = Image.open(img_name).convert('L')  # grayscale
        label = self.label_map[self.data.iloc[idx, 2]]

        if self.transform:
            image = self.transform(image)

        return image, label
