# Author: Jose G. Perez

import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nni.nas.pytorch.callbacks import LRSchedulerCallback
from nni.nas.pytorch.callbacks import ModelCheckpoint
from nni.algorithms.nas.pytorch.spos import SPOSSupernetTrainingMutator, SPOSSupernetTrainer
from xray_dataset import XrayImageDataset
from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import CrossEntropyLabelSmooth, accuracy_topk

logger = logging.getLogger("nni.spos.supernet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--dataset-dir", type=str, default="/data/datasets/xray-dataset/v2/")
    parser.add_argument("--load-checkpoint", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--debugging", action="store_true", default=False)

    args = parser.parse_args()

    # Using a fixed set of image will improve the performance
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    assert torch.cuda.is_available()

    # Prepare data and model
    train_data, val_data, test_data, input_shape, num_classes = XrayImageDataset.get_datasets(args.dataset_dir, args.debugging)
    model = ShuffleNetV2OneShot(input_size=input_shape[1], n_classes=num_classes, op_flops_path='./data/flops.pkl')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        model.cuda()

    # Load previous checkpoint if needed
    if args.load_checkpoint:
        model.load_state_dict(load_and_parse_state_dict())

    # Prepare all things required by the trainer
    mutator = SPOSSupernetTrainingMutator(model,
                                          flops_func=model.get_candidate_flops,
                                          flops_lb=290E6,
                                          flops_ub=360E6)
    criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs) if step <= args.epochs else 0,
                                                  last_epoch=-1)

    # Prepare data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    # Prepare trainer
    trainer = SPOSSupernetTrainer(model, criterion, accuracy_topk, optimizer,
                                  args.epochs, train_loader, val_loader,
                                  mutator=mutator, batch_size=args.batch_size,
                                  log_frequency=args.log_frequency, workers=args.workers,
                                  callbacks=[LRSchedulerCallback(scheduler),
                                             ModelCheckpoint("./checkpoints")])
    trainer.train()
