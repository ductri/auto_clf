import logging
from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset

voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class PositiveDataset(Dataset):
    def __init__(self, path_to_file):
        super(PositiveDataset, self).__init__()

        df = pd.read_csv(path_to_file, usecols=['mention'])
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        self.mentions = list(df['mention'])

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx]

