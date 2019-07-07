import logging

import numpy as np

from torch.utils.data import DataLoader
import torch
from naruto_skills.new_voc import Voc

# from data_for_train.not_shared_pool import PoolDocs
from data_for_train.shared_pool import PoolDocs
from data_for_train.index_dataset import IndexDataset
from data_for_train.positive_dataset import PositiveDataset
from data_for_train.full_dataset import FullDataset


MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'
voc = Voc.load('/source/main/vocab/output/voc.pkl')


def create_data_loader(batch_size, num_workers, shuffle=True):
    def collate_fn(list_data):
        """
        shape == (batch_size, col1, col2, ...)
        """
        data = zip(*list_data)
        data = [np.stack(col, axis=0) for col in data]
        data = [torch.from_numpy(col) for col in data]
        return data

    dataset_dir = '/source/main/data_for_train/output/train/pool'
    pool = PoolDocs(dataset_dir)
    pool = IndexDataset(voc, pool, equal_length=MAX_LENGTH)
    positive_data = PositiveDataset('/source/main/data_for_train/output/train/positive_class_1.csv')
    positive_data = IndexDataset(voc, positive_data, equal_length=MAX_LENGTH)
    ds = FullDataset(pool, positive_data)
    logging.info('Positive/Total: %s/%s', int(sum(ds.labels)), len(ds))

    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dl
