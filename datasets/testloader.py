import os
import math
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, config, frame, transform=None):
        self.config = config
        self.frame = frame
        self.transform = transform

    def __len__(self):
        return self.frame.shape[0]

    def __getitem__(self, idx):
        fname = os.path.splitext(self.frame['fname'][idx])[0] + '.pkl'
        fpath = os.path.join(self.config.DATA_DIR, self.frame['type'][idx], fname)

        data = pickle.load(open(fpath, "rb"), encoding='latin1')

        data = self._window_stack(data)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def _window_stack(self, data):
        input_length = int(self.config.DATA.N_MELS * self.config.DATA.DURATION)
        n_windows = math.ceil(data.shape[2] / input_length)

        batch = np.zeros((n_windows, data.shape[0], data.shape[1], input_length), np.float32)

        if data.shape[2] <= input_length:
            offset = (input_length - data.shape[2]) // 2
            data = np.pad(data, ((0, 0), (0, 0), (offset, input_length - data.shape[2] - offset)), "constant")
            batch[0] = data

        else:
            for i in range(n_windows - 1):
                batch[i] = data[:, :, input_length*i:input_length*(i + 1)]

            last_window = data[:, :, input_length*(n_windows-1):input_length*n_windows]
            offset = (input_length - last_window.shape[2]) // 2
            last_window = np.pad(last_window, ((0, 0), (0, 0), (offset, input_length - last_window.shape[2] - offset)), "constant")
            batch[n_windows-1] = last_window

        return batch


class TestLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.size = len(self.dataset)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration

        data = self.dataset[self.index]
        self.index += 1

        return data

    def __len__(self):
        return self.size


def get_testloader(config, frame, transform=None):
    testset = TestDataset(config, frame, transform)
    testloader = TestLoader(testset)

    return testloader
