import ast

from tqdm import tqdm
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

from preprocess import preprocessor


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    root = Path('/source/main/data_download/output/topics')
    total_rows = 0
    for idx, p in tqdm(list(enumerate(root.glob('*.csv')))):
        df = pd.read_csv(str(p))
        if df.shape[0] == 0:
            continue
        df['mention'] = df['search_text'].map(lambda x: ast.literal_eval(x)[1])
        df['mention'] = Parallel(n_jobs=-1)(delayed(preprocessor.train_preprocess)(doc, 100) for doc in list(df['mention']))

        df.to_csv('/source/main/data_for_train/output/huge_pool//topics/%s' % p.name, index=None)
        total_rows += df.shape[0]
        if idx % 10 == 0:
            print('Total rows: %s' % total_rows)
