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
    def __init__(self, dataset_dir, max_size=9e5):
        super(PoolDocs, self).__init__()

        path = Path(dataset_dir)
        # wiki_path = path/'wiki.csv'
        social_path = path.glob('*.csv')

        # df = pd.read_csv(str(wiki_path), nrows=3e5, usecols=['target'])
        # df.rename(columns={'target': 'mention'}, inplace=True)

        df = pd.DataFrame()
        preserved_topics = set(topic_ids.test_topics)
        for topic_path in social_path:
            if topic_path.name.split('.')[0] in preserved_topics:
                logging.info('Topic %s is preserved for testing', topic_path.name.split('.')[0])
                continue
            df_topic = pd.read_csv(topic_path, nrows=5e6, usecols=['mention', 'mention_type'])
            df_topic = df_topic[df_topic['mention_type'] != 3]
            df = df.append(df_topic)

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        logging.info('Temp total: %s', df.shape[0])
        self.mentions = list(df['mention'].sample(int(max_size), random_state=RANDOM_STATE, replace=True).drop_duplicates())

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx]

