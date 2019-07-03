import logging
import json
import ast
from pathlib import Path

import pandas as pd


MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class SolrTopics:
    def __init__(self, dataset_dir):
        super(SolrTopics, self).__init__()

        path = Path(dataset_dir)
        social_path = path.glob('*.csv')

        df = pd.DataFrame()
        for topic_path in social_path:
            df_topic = pd.read_csv(topic_path, usecols=['id', 'topic_id', 'search_text', 'mention_type'])
            df_topic.dropna(inplace=True)
            df_topic.drop_duplicates(inplace=True)
            df_topic = df_topic['']
            df_topic['search_text'] = df_topic['search_text'].map(ast.literal_eval)
            df = df.append(df_topic)

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        self.mentions = list(df['mention'])

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx]

