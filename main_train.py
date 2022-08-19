#!/usr/bin/env python3
import timm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
from data_transform import data_transforms
import torch.nn as nn
import util_logger as L
import argparse
import warnings
import logging
import torch
import copy
import os
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Intel Image Classification')
    parser.add_argument("--bsize", type=int, help="batch size", default=32)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--saveModel",  action='store_true', help="if to save model")
    parser.add_argument("--load_ckpt", type=int, help="load checkpoint from model zoo", default=0)
    parser.add_argument("--pretrained", action='store_true', help="pretrain or not")
    parser.add_argument("--model_name", type=str, help="model name", default='inception_v4')
    args = parser.parse_args()
    return args


def train_model(logger, model, optimizer, criterion, scheduler, num_epochs, begin, suffix):
    """
    :param logger: logger to save info
    :param model: classifier model
    :param optimizer: optimizer(Adam or others)
    :param criterion: cross entropy or others
    :param scheduler: learning scheduler
    :param num_epochs: train epochs
    :param begin: the check point the begin
    :param suffix: load model ckpt suffix
    :return: best test model
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(begin+1, num_epochs+begin+1):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs+begin))
        logger.info('-' * 50)

        for x in ['train', 'test']:
            if x == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = len(dataloaders[x])
            for i, (inputs, labels) in enumerate(dataloaders[x]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(x == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if x == 'train':
                        loss.backward()
                        optimizer.step()

                loss_step = loss.item() * inputs.size(0)
                corrects = torch.sum(preds == labels.data)
                if i % 50 == 0:
                    logger.info("Step:{}/{}, Loss: {:.4f}, Corrects: {}"
                                .format(i, total, loss_step, corrects))

                running_loss += loss_step
                running_corrects += corrects

            if x == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[x]
            epoch_acc = running_corrects.double() / dataset_sizes[x]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(x, epoch_loss, epoch_acc))

            if epoch % 5 == 0:
                torch.save(best_model_wts, os.path.join(model_dir, '{}_{}.pth'.format(args.model_name, epoch)))

            if x == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('-' * 30)

    logger.info('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    torch.save(best_model_wts, os.path.join(model_dir, args.model_name+'_'+suffix+'.pth'))

    return model


if __name__ == '__main__':
    args = parse_args()
    data_dir = './data/archive'
    result_dir = './results'
    model_dir = './model_zoo'
    train_log_file = args.model_name + '_train.log'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logger
    L.logger_info(train_log_file, log_path=os.path.join(result_dir, train_log_file))
    logger_train = logging.getLogger(train_log_file)

    # dataset folder
    phase = ['train', 'test']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, 'seg_'+x), data_transforms[x])
                      for x in phase}
    # dataloader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bsize,
                                                  shuffle=True,num_workers=4)
                   for x in phase}

    dataset_sizes = {x: len(image_datasets[x]) for x in phase}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use device: {}".format(device))
    # model definition --own model modified based on ConvMixer
    if args.model_name == 'MConvMixer':
        from networks import MConvMixer
        model = MConvMixer.MConvMixer(dims=(384, 768, 1536), depths=(6, 6, 8),
                 patch_size=(7, 7), n_class=6)
    else:
        model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=6)

    model.to(device)
    # if use pretrained model, the suffix becomes 'best' other wise 'bestn'
    suffix = 'best' if args.pretrained else 'bestn'
    lr = 6e-5 if args.pretrained else 5e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': lr}],
                           lr=lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=args.load_ckpt)

    for p in model.parameters():
        p.requires_grad = True

    if args.load_ckpt != 0:
        state_dict = torch.load(os.path.join(model_dir,
                                             '{}_{}.pth'.format(args.model_name,args.load_ckpt)))
        model.load_state_dict(state_dict)
    logger_train.info("Train model with {} pretrained".format('' if args.pretrained else 'NO'))

    train_model(logger_train, model, optimizer, criterion, scheduler,
                args.epochs, begin=args.load_ckpt, suffix=suffix)

