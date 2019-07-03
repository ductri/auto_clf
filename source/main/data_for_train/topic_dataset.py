import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from data_download import topic_ids

voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class Topic(Dataset):
    def __init__(self, path_to_file):
        super(Topic, self).__init__()
        df_topic = pd.read_csv(path_to_file, usecols=['mention', 'mention_type'])

        df_topic.dropna(inplace=True)
        df_topic.drop_duplicates(inplace=True)
        self.mentions = list(df_topic['mention'])

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx]

