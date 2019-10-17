import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def binary_focal_loss(gamma=2):
    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func


def bce():
    return nn.BCEWithLogitsLoss()


def get_loss(config):
    print('loss name:', config.LOSS.NAME)
    f = globals().get(config.LOSS.NAME)
    if config.LOSS.PARAMS is None:
        return f()
    else:
        return f(**config.LOSS.PARAMS)
