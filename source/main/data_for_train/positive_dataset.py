import logging
from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset

voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class PositiveDataset(Dataset):
    def __init__(self, list_data):
        super(PositiveDataset, self).__init__()
        self.mentions = list(list_data)

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx], 1


class PositiveDataset2(Dataset):
    def __init__(self, list_data):
        super(PositiveDataset2, self).__init__()
        self.mentions = list(list_data)
        self.__size = len(self.mentions)

    def __len__(self):
        return 794323

    def __getitem__(self, idx):
        return self.mentions[idx % self.__size], 1
