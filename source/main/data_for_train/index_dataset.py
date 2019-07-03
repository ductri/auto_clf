from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, voc, text_ds, equal_length):
        super(IndexDataset, self).__init__()

        self.docs = voc.docs2idx(text_ds, equal_length=equal_length)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]
