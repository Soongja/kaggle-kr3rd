import os
import cv2
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.tools import one_hot_encoding


class CarDataset(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        frame = pd.read_csv(self.config.CSV)
        self.frame = frame.loc[frame['split'] == self.split].reset_index(drop=True)
        if self.config.DEBUG:
            self.frame = self.frame[:100]
        print(self.split, 'set:', self.frame.shape[0])

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        dir_split = 'train' if self.split in ['train', 'val'] else 'test'
        image = cv2.imread(os.path.join(self.config.DATA_DIR, dir_split, self.frame["img_file"][idx]), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop
        # x1, y1, x2, y2 = self.frame.iloc[idx][1:5]
        # image = image[y1:y2 + 1, x1:x2 + 1]

        # pad
        if image.shape[0] / image.shape[1] > 0.536:
            w_offset = int(((image.shape[0] / 0.536) - image.shape[1]) / 2)
            image = np.pad(image, ((0, 0), (w_offset, w_offset), (0, 0)), 'constant', constant_values=(0))
        else:
            h_offset = int(((image.shape[1] * 0.536) - image.shape[0]) / 2)
            image = np.pad(image, ((h_offset, h_offset), (0, 0), (0, 0)), 'constant', constant_values=(0))

        image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        if self.transform is not None:
            image = self.transform(image)

        if not self.split == 'test':
            label = one_hot_encoding(self.frame["class"][idx] - 1)
            label = torch.from_numpy(label).float()
            return image, label
        else:
            return image
