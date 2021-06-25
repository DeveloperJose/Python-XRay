import argparse
import logging
import random
import time
import joblib
from itertools import cycle

import nni
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nni.algorithms.nas.pytorch.classic_nas import get_and_apply_next_architecture
from nni.nas.pytorch.utils import AverageMeterGroup

from xray_dataset import XrayImageDataset
from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import CrossEntropyLabelSmooth, accuracy_topk, get_flops

logger = logging.getLogger("nni.spos.tester")


def retrain_bn(model, criterion, max_iters, log_freq, loader):
    with torch.no_grad():
        logger.info("Clear BN statistics...")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        logger.info("Train BN with training set (BN sanitize)...")
        model.train()
        meters = AverageMeterGroup()
        for step in range(max_iters):
            inputs, targets = next(loader)
            # Send to GPU
            targets = targets.cuda()
            inputs = inputs.cuda()

            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy_topk(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == max_iters:
                logger.info("Train Step [%d/%d] %s", step + 1, max_iters, meters)


def test_acc(model, criterion, log_freq, loader):
    logger.info("Start testing...")
    model.eval()
    meters = AverageMeterGroup()
    start_time = time.time()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            # Send from CPU to GPU
            targets = targets.cuda()
            inputs = inputs.cuda()

            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy_topk(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == len(loader):
                logger.info("Valid Step [%d/%d] time %.3fs acc1 %.4f loss %.4f",
                            step + 1, len(loader), time.time() - start_time,
                            meters.acc1.avg, meters.loss.avg)
    return meters.acc1.avg


def evaluate_acc(model, criterion, args, loader_train, loader_test):
    acc_before = test_acc(model, criterion, args.log_frequency, loader_test)
    nni.report_intermediate_result(acc_before)

    retrain_bn(model, criterion, args.train_iters, args.log_frequency, loader_train)
    acc = test_acc(model, criterion, args.log_frequency, loader_test)
    assert isinstance(acc, float)
    nni.report_intermediate_result(acc)
    nni.report_final_result(acc)


def main():
    parser = argparse.ArgumentParser("SPOS Candidate Tester")
    parser.add_argument("--dataset-dir", type=str, default="/data/datasets/xray-dataset/v2/")
    parser.add_argument("--checkpoint", type=str, default="./data/checkpoint-150000.pth.tar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-iters", type=int, default=200)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--debugging", action="store_true", default=False)
    parser.add_argument("--generate-flops", action="store_true", default=False)
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
    #model = ShuffleNetV2OneShot(input_size=input_shape[1], n_classes=num_classes, op_flops_path='./data/op_flops_dict.pkl')
    model = ShuffleNetV2OneShot(op_flops_path='./data/op_flops_dict.pkl')
    criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1)
    logger.info(f"CUDA Available={torch.cuda.is_available()}, Initialized={torch.cuda.is_initialized()}, Device_Count: {torch.cuda.device_count()}")
    using_cuda = False
    if torch.cuda.device_count() >= 1:
        model = nn.DataParallel(model)
        model.cuda()
        logger.info(f"Using CUDA: {model.device_ids}")
        using_cuda = True

    # Generate flops file if asked
    if args.generate_flops:
        logger.info("Creating flops PKL")
        flops = get_flops(model, input_shape=input_shape)
        joblib.dump(flops, './data/flops.pkl')
        return

    # Load checkpoint
    get_and_apply_next_architecture(model)
    checkpoint_dict = load_and_parse_state_dict(filepath=args.checkpoint, using_cuda=using_cuda)
    #logger.debug("Checkpoint dictionary: ")
    #logger.debug(checkpoint_dict.keys())
    model.load_state_dict(checkpoint_dict)

    # Prepare data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    train_loader = cycle(train_loader)

    # Begin evaluation
    evaluate_acc(model, criterion, args, train_loader, val_loader)


if __name__ == "__main__":
    main()
