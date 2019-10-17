import numpy as np
import pandas as pd

# for i in range(5):
train_frame = pd.read_csv('stratified/5fold_4.csv')
pseudo = pd.read_csv('se_resnext_50_256_480_5fold_all.csv')
# print(train_frame.head())
# print(pseudo.head())

pseudo = pseudo[pseudo['confidence']>0.50]
# print(pseudo.head())

pseudo['split'] = 'train'
pseudo = pseudo[['img_file', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'split']]
# print(pseudo.head())

out = pd.concat([train_frame, pseudo], ignore_index=True)
# print(out.head())

out.to_csv('ensemble_pseudo0.50+5fold_4.csv', index=False)
