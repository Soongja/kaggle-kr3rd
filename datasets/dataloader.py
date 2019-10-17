from torch.utils.data import DataLoader

from .datasets import CarDataset


def get_dataset(config, split, transform):
    f = globals().get(config.DATA.NAME)

    return f(config, split, transform)


def get_dataloader(config, split, transform=None):
    dataset = get_dataset(config, split, transform)

    is_train = 'train' == split
    batch_size = config.TRAIN.BATCH_SIZE if is_train else config.EVAL.BATCH_SIZE

    dataloader = DataLoader(dataset,
                             shuffle=is_train,
                             batch_size=batch_size,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True)

    return dataloader
