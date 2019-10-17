import os
import shutil
import random
import cv2
import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from models.model_factory import get_model
from factory.losses import get_loss
from factory.schedulers import get_scheduler
from factory.optimizers import get_optimizer
from factory.transforms import Normalize, ToTensor, Albu
from datasets.dataloader import get_dataloader

import utils.config
import utils.checkpoint
from utils.metrics import batch_f1_score
from utils.tools import prepare_train_directories, AverageMeter, one_hot_encoding, mixup_data


def evaluate_single_epoch(config, model, dataloader, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)

            loss = criterion(logits, labels)
            losses.update(loss.item(), images.shape[0])

            score = batch_f1_score(logits, labels)
            scores.update(score.item(), images.shape[0])

            del images, labels, logits
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()

            print('[%2d/%2d] time: %.2f, val_loss: %.6f, val_score: %.4f'
                  % (i, len(dataloader), batch_time.sum, loss, score))

        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/score', scores.avg, epoch)

    return scores.avg


def train_single_epoch(config, model, dataloader, criterion, optimizer, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        labels = labels.cuda()

        if config.MIXUP:
            images, labels = mixup_data(images, labels)

        logits = model(images)

        loss = criterion(logits, labels)
        losses.update(loss.item(), images.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = batch_f1_score(logits, labels, mixup=config.MIXUP)
        scores.update(score.item(), images.shape[0])

        if i == 0:
            print(labels[0])
            print(np.unique(labels.detach().cpu().numpy()[0]))
            img = images.detach().cpu().numpy()[0]
            img = np.transpose(img, (1, 2, 0))
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.uint8(img * 255)
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        del images, labels, logits
        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()

        print("[%d/%d][%d/%d] time: %.2f, train_loss: %.6f, train_score: %.4f, lr: %f"
              % (epoch, config.TRAIN.NUM_EPOCHS, i, len(dataloader), batch_time.sum, loss, score, optimizer.param_groups[0]['lr']))

    writer.add_scalar('train/score', scores.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)


def train(config, model, train_loader, test_loader, criterion, optimizer, scheduler, writer, start_epoch, best_score):
    num_epochs = config.TRAIN.NUM_EPOCHS
    model = model.cuda()

    for epoch in range(start_epoch, num_epochs):
        train_single_epoch(config, model, train_loader, criterion, optimizer, writer, epoch)

        test_score = evaluate_single_epoch(config, model, test_loader, criterion, writer, epoch)

        print('Total Test Score: %.4f' % test_score)
        if test_score > best_score:
            best_score = test_score
            print('Test score Improved! Save checkpoint')
            utils.checkpoint.save_checkpoint(config, model, epoch, test_score)

        if config.SCHEDULER.NAME == 'reduce_lr_on_plateau':
            scheduler.step(test_score)
        else:
            scheduler.step()


def run(config):
    model = get_model(config).cuda()
    criterion = get_loss(config)
    optimizer = get_optimizer(config, model.parameters())

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score = utils.checkpoint.load_checkpoint(config, model, checkpoint)
    else:
        print('[*] no checkpoint found')
        last_epoch, score = -1, -1

    print('last epoch:{} score:{:.4f}'.format(last_epoch, score))

    optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.LR
    scheduler = get_scheduler(config, optimizer, last_epoch)
    if last_epoch != -1:
        scheduler.step()

    writer = SummaryWriter(os.path.join(config.TRAIN_DIR, 'logs'))

    train_loader = get_dataloader(config, 'train', transform=transforms.Compose([Albu(),
                                                                                 Normalize(),
                                                                                 ToTensor()]))
    test_loader = get_dataloader(config, 'val', transform=transforms.Compose([Normalize(),
                                                                               ToTensor()]))

    train(config, model, train_loader, test_loader, criterion, optimizer, scheduler, writer, last_epoch+1, score)


def seed_everything():
    seed = 2019
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('start training.')
    seed_everything()

    yml = 'configs/base.yml'
    config = utils.config.load(yml)
    prepare_train_directories(config)
    pprint.pprint(config, indent=2)
    shutil.copy(yml, os.path.join(config.TRAIN_DIR, 'config.yml'))

    run(config)
    print('success!')


if __name__ == '__main__':
    main()
