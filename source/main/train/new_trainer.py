import logging
import time

import torch
import numpy as np
import pandas as pd
from naruto_skills.new_voc import Voc
from torch.utils.data import DataLoader, Dataset
from naruto_skills.dl_logging import DLTBHandler, DLLoggingHandler, DLLogger
from tqdm import tqdm

from model_def.siamese_model_8 import SiameseModel
from model_def.siamese_core import SiameseModelCore
from data_for_train.index_dataset import IndexDataset
from naruto_skills.training_checker import TrainingChecker


class TripletDataset(Dataset):
    def __init__(self, pos, pool):
        super(TripletDataset, self).__init__()
        self.pos = pos
        self.pool = pool
        self.len_pos = len(pos)

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx):
        return self.pos[idx % self.len_pos], self.pool[idx]


def prepare_dataset(name):
    def collate_fn(list_data):
        """
        shape == (batch_size, col1, col2, ...)
        """
        data = zip(*list_data)
        data = [np.stack(col, axis=0) for col in data]
        data = [torch.from_numpy(col) for col in data]
        return data
    MAX_LENGTH = 100
    BATCH_SIZE = 100
    df_pool = pd.read_csv('/source/main/data_for_train/output/%s/pool.csv' % name, nrows=1e6)
    df_pool.dropna(inplace=True, subset=['mention'])
    df_pool.drop_duplicates(inplace=True, subset=['mention'])
    df_pool = df_pool.iloc[:794323, :]
    pool = IndexDataset(voc, list(df_pool['mention']), equal_length=MAX_LENGTH)

    POSITIVE_NAME = 'positive_class_1'
    df_pos = pd.read_csv('/source/main/data_for_train/output/%s/%s.csv' % (name, POSITIVE_NAME))
    df_pos.dropna(inplace=True, subset=['mention'])
    df_pos.drop_duplicates(inplace=True, subset=['mention'])

    logging.info(df_pos.shape)
    pos = IndexDataset(voc, list(df_pos['mention']), equal_length=MAX_LENGTH)

    ds = TripletDataset(pos, pool)
    data_loader = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    return data_loader, pos, pool


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root_dir = '/source/main/train/output/'
    # experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
    experiment_name = '15.1'

    # Dataset prepare
    voc = Voc.load('/source/main/vocab/output/voc.pkl')
    train_loader, _, _ = prepare_dataset('train')
    eval_loader, eval_pos, eval_pool = prepare_dataset('eval')

    core_model = SiameseModelCore(voc.get_embedding_weights())
    model = SiameseModel(core_model)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.build_stuff_for_training(device)

    train_logger = DLLogger()
    train_logger.add_handler(DLLoggingHandler())
    train_logger.add_handler(DLTBHandler(root_dir + '/' + 'logging/' + experiment_name + '/train'))

    eval_logger = DLLogger()
    eval_logger.add_handler(DLLoggingHandler())
    eval_logger.add_handler(DLTBHandler(root_dir + '/' + 'logging/' + experiment_name + '/eval'))

    training_checker = TrainingChecker(model, root_dir=root_dir+'/saved_models/' + experiment_name, init_score=-10000)

    step = 0
    num_epochs = 1
    total = num_epochs * len(train_loader)
    for epoch_idx in range(num_epochs):
        for inputs in train_loader:
            start = time.time()
            step += 1
            inputs = [i.to(device) for i in inputs]
            l = model.train_batch(inputs[0], inputs[1])

            if step % 10 == 0:
                train_logger.handlers[0].add_scalar('progress', step / total, step)
                train_logger.add_scalar('train/loss', l, step)
                train_logger.add_scalar('train/duration', time.time() - start, step)

            if step % 200 == 0:
                eval_start = time.time()
                model.eval()
                with torch.no_grad():
                    eval_losses = [model.get_loss(inputs[0].to(device), inputs[1].to(device)) for inputs in tqdm(eval_loader)]
                    eval_losses = [v.item() for v in eval_losses]
                    eval_loss_mean = np.mean(eval_losses)
                    eval_logger.add_scalar('eval/loss', eval_loss_mean, step)
                    eval_logger.add_scalar('eval/loss_std', np.std(eval_losses), step)
                    eval_logger.add_scalar('eval/duration', time.time()-eval_start, step)
                    training_checker.update(-eval_loss_mean, step)

    print(training_checker.best())
    training_checker.update(0, step)
