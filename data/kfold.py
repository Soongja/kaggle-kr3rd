import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

frame = pd.read_csv('train.csv')

x = frame['img_file'].values
y = frame['class'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1992)
skf.get_n_splits(x, y)

frame['split'] = 'default'

for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
    new_frame = frame
    # if fold == 0:
    print(fold, len(train_index), len(val_index))
    new_frame['split'].iloc[train_index] = 'train'
    new_frame['split'].iloc[val_index] = 'val'
    print(new_frame.head())
    frame.to_csv('stratified/5fold_%d.csv' % fold, index=False)
