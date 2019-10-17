import random
import numpy as np
import torch
import cv2
from albumentations import (
    OneOf, Compose, HorizontalFlip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur,
    CLAHE, IAASharpen, GaussNoise, IAAAdditiveGaussianNoise,
    HueSaturationValue, RGBShift, ChannelShuffle,
    ToGray, RandomSizedCrop)


class Normalize:
    """
    normalize data to -1 ~ 1
    """
    def __call__(self, data):
        smooth = 1e-6

        # data = (data - np.min(data) + smooth) / (np.max(data) - np.min(data) + smooth)
        data = data / 255.0
        data = data * 2 - 1

        return data


class ToTensor:
    """
    convert ndarrays to Tensors.
    """
    def __call__(self, data):
        data = np.transpose(data, (2, 0, 1))
        data = torch.from_numpy(data).float()
        return data


def strong_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(p=0.75, shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=1.0),
        OneOf([
            HueSaturationValue(p=1.0),
            RGBShift(p=1.0),
            ChannelShuffle(p=1.0)
        ], p=1.0),
        OneOf([
            Blur(p=1.0),
            MedianBlur(p=1.0),
            MotionBlur(p=1.0),
        ], p=0.5),
        OneOf([
            GridDistortion(p=1.0),
            ElasticTransform(p=1.0)
        ], p=0.5),
        OneOf([
            CLAHE(p=1.0),
            IAASharpen(p=1.0),
        ], p=0.5),
        GaussNoise(p=0.5)
        # ToGray(p=1.0),
    ], p=p)


def medium_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(p=0.75, shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.5),
        OneOf([
            HueSaturationValue(p=1.0),
            RGBShift(p=1.0),
            ChannelShuffle(p=1.0)
        ], p=0.5),
        OneOf([
            Blur(p=1.0),
            MedianBlur(p=1.0),
            MotionBlur(p=1.0),
        ], p=0.3),
        OneOf([
            GridDistortion(p=1.0),
            ElasticTransform(p=1.0)
        ], p=0.3),
        OneOf([
            CLAHE(p=1.0),
            IAASharpen(p=1.0),
        ], p=0.3),
        IAAAdditiveGaussianNoise(p=0.5)
        # ToGray(p=1.0),
    ], p=p)


def weak_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            RandomBrightness(limit=0.2, p=1.0),
            RandomContrast(limit=0.2, p=1.0)
        ], p=0.5),
        ShiftScaleRotate(p=0.4, shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
        IAAAdditiveGaussianNoise(p=0.3)
    ], p=p)


class Albu():
    def __call__(self, image):
        # augmentation = strong_aug()
        augmentation = weak_aug()

        data = {"image": image}
        augmented = augmentation(**data)

        return augmented["image"]


class TestNormalize:
    def __call__(self, data):
        smooth = 1e-6

        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                data[b][c] = (data[b][c] - np.min(data[b][c]) + smooth) / (np.max(data[b][c]) - np.min(data[b][c]) + smooth)

        data = data * 2 - 1

        return data


if __name__=="__main__":
    img = cv2.imread('train_00008.jpg', 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (380, 256))

    for i in range(100):
        data = {"image": img}
        # aug = strong_aug()
        aug = weak_aug()
        # aug = medium_aug()
        augmented = aug(**data)

        out_img = augmented["image"]
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('img', out_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
