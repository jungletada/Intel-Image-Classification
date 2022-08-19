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
import tqdm
import os

warnings.filterwarnings("ignore")
cls = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:"street"}

def parse_args():
    parser = argparse.ArgumentParser(description='Intel Image Classification')
    parser.add_argument("--bsize", type=int, help="batch size", default=32)
    parser.add_argument("--pretrained", action='store_true', help="pretrain or not")
    parser.add_argument("--model_name", type=str, help="model name", default='vit_base_patch16_224')
    args = parser.parse_args()
    return args


def append_res(logger, names, preds):
    with open(logger, 'a') as l:
        for i in range(len(names)):
            pred = int(preds[i].item())
            res = "{},{}".format(names[i], cls[pred])
            l.write(res)


def pred_model():
    args = parse_args()
    data_dir = './data/archive'
    result_dir = './results'
    model_dir = './model_zoo'
    pred_log_file = args.model_name + '_pred.log'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    from data_transform import PredDataset
    pred_dataset = PredDataset(os.path.join(data_dir, 'pred'))
    L.logger_info(pred_log_file, log_path=os.path.join(result_dir, pred_log_file))
    logger_pred = logging.getLogger(pred_log_file)

    dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=args.bsize,
                                             shuffle=False, num_workers=4)

    if args.model_name == 'MConvMixer':
        from networks import MConvMixer
        model = MConvMixer.MConvMixer(dims=(384, 768, 1536), depths=(6, 6, 8),
                                      patch_size=(7, 7), n_class=6)
        state_dict = torch.load(os.path.join(model_dir,
                                             '{}_{}.pth'.format(args.model_name, args.load_ckpt)))

    else:
        model = timm.create_model(args.model_name, pretrained=False, num_classes=6)
        state_dict = torch.load(os.path.join(model_dir,
                                             '{}_{}.pth'.format(args.model_name, 'best')))

    model.load_state_dict(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    logger_pred.info("total parameters: {:.2f}M".format(total_params / 1e6))
    model.to(device)
    
    tloader = tqdm.tqdm(dataloader)
    for _, inputs in enumerate(tloader):
        imgs = inputs['img'].to(device)
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            append_res(logger_pred, inputs['name'], preds)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use device: {}".format(device))
    pred_model()
