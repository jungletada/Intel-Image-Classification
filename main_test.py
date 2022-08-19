#!/usr/bin/env python3
import timm
from torchvision import datasets
from data_transform import data_transforms
import util_logger as L
import argparse
import warnings
import logging
import torch
import os
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Intel Image Classification')
    parser.add_argument("--bsize", type=int, help="batch size", default=32)
    parser.add_argument("--pretrained", action='store_true', help="pretrain or not")
    parser.add_argument("--model_name", type=str, help="model name", default='inception_v4')
    args = parser.parse_args()
    return args


def validation_model_only(logger, model):
    """
    :param logger: logger to save info
    :param model: classifier model
    :return:
    """
    from time import time
    running_corrects = 0
    model.eval()
    begin = time()
    for i, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        corrects = torch.sum(preds == labels.data)
        running_corrects += corrects
    end = time()
    running_time = end-begin
    logger.info('running time: {:.4f}s'.format(running_time))
    epoch_acc = running_corrects.double() / dataset_sizes
    logger.info('Acc: {:.4f}'.format(epoch_acc))
    logger.info('-' * 50)

if __name__ == '__main__':
    args = parse_args()
    data_dir = './data/archive'
    result_dir = './results'
    model_dir = './model_zoo'
    test_log_file = args.model_name + '_test.log'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # logger
    L.logger_info(test_log_file, log_path=os.path.join(result_dir, test_log_file))
    logger_test = logging.getLogger(test_log_file)

    # dataset folder
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'seg_test'), data_transforms['test'])

    # dataloader
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=args.bsize,
                                              shuffle=True, num_workers=4)

    dataset_sizes = len(image_datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use device: {}".format(device))

    # model definition --own model modified based on ConvMixer
    if args.model_name == 'MConvMixer':
        from networks import MConvMixer
        model = MConvMixer.MConvMixer(dims=(384, 768, 1536), depths=(6, 6, 8),
                                      patch_size=(7, 7), n_class=6)
    else:
        model = timm.create_model(args.model_name, pretrained=False,
                                  num_classes=6)
    # if use pretrained model, the suffix becomes 'best' other wise 'bestn'
    suffix = 'best' if args.pretrained else 'bestn'
    total_params = sum(p.numel() for p in model.parameters())
    print("total parameters: {:.2f}M".format(total_params / 1e6))
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    state_dict = torch.load(os.path.join(model_dir, '{}_{}.pth'.format(args.model_name, suffix)))

    model.load_state_dict(state_dict)
    logger_test.info("Test model with {} pretrained".format('' if args.pretrained else 'NO'))
    validation_model_only(logger_test, model)
