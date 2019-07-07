import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from data_download import topic_ids

voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'

RANDOM_STATE = 42


class PoolDocs(Dataset):
    def __init__(self, list_mentions):
        super(PoolDocs, self).__init__()
        self.mentions = list_mentions
        logging.info('Total: %s', len(self.mentions))

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx], 0

