
import numpy as np

from torch.utils.data import Dataset, Subset


voc = None
MAX_LENGTH = 100
NUM_WORKERS = 0
ROOT = '/source/'


class FullDataset(Dataset):
    def __init__(self, pool_ds, positive_ds):
        super(FullDataset, self).__init__()

        # anchor_size = 100
        # spy_ds = Subset(positive_ds, indices=range(anchor_size))
        # positive_ds = Subset(positive_ds, indices=range(anchor_size, len(positive_ds)))

        # negative_ds = pool_ds + spy_ds
        self.mentions = positive_ds + pool_ds
        self.labels = list(np.concatenate((np.ones(len(positive_ds)), np.zeros(len(pool_ds))), axis=0).astype(int))

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        return self.mentions[idx], self.labels[idx]


