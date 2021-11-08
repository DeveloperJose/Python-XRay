#####################################################
# Author: Jose G. Perez
# Modified from AutoDL-Projects/exps/basic/basic-main.py (by Xuanyi Dong [GitHub D-X-Y])
#####################################################
import sys, time, torch, random, argparse
from PIL import ImageFile
from torchvision.transforms import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

from xautodl.procedures import (

    prepare_seed,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.procedures import get_optim_scheduler, get_procedures
from xautodl.models import obtain_model
from xautodl.nas_infer_model import obtain_nas_infer_model
from xautodl.utils import get_model_infos
from xautodl.log_utils import Logger, AverageMeter, time_string, convert_secs2time
from xautodl.config_utils import load_config

from nni.xray_dataset import XrayImageDataset

def main():
    RAND_SEED = 42
    BATCH_SIZE = 128
    N_WORKERS = 12
    PRINT_FREQUENCY = 100
    PRINT_FREQUENCY_EVAL = 200
    EVAL_FREQUENCY = 1
    PROCEDURE = 'basic'

    # The type of model
    MODEL_SOURCE = 'normal'
    # Extra model CKP file (help to indicate searched architecture)
    EXTRA_MODEL_PATH = None
    # Resume path (if exists)
    RESUME_PATH = './output/model2/checkpoint/seed-42-basic.pth'
    # The path of the initialization model
    INIT_MODEL_PATH = None
    # The path for the model architecture configuration
    MODEL_ARCH_CONFIG_PATH = 'XRAY-Arch.config'
    # The path for the model optimization configuration
    MODEL_OPT_CONFIG_PATH = 'XRAY-Opts.config'
    # Directory to save log and model files
    SAVE_DIR = './output/round3/'

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.set_num_threads(N_WORKERS)

    prepare_seed(RAND_SEED)
    logger = Logger(SAVE_DIR, RAND_SEED)

    train_data, val_data, test_data, input_shape, num_classes = XrayImageDataset.get_datasets("/data/datasets/xray-dataset/v3/", debugging=False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=True,
    )

    # Get model and optimizer configurations
    model_config = load_config(MODEL_ARCH_CONFIG_PATH, {"class_num": num_classes}, logger)
    optim_config = load_config(MODEL_OPT_CONFIG_PATH, {"class_num": num_classes}, logger)

    # Obtain the model based on the model source type
    if MODEL_SOURCE == "normal":
        base_model = obtain_model(model_config)
    elif MODEL_SOURCE == "nas":
        base_model = obtain_nas_infer_model(model_config, EXTRA_MODEL_PATH)
    elif MODEL_SOURCE == "autodl-searched":
        base_model = obtain_model(model_config, EXTRA_MODEL_PATH)
    else:
        raise ValueError(f"invalid model-source : {MODEL_SOURCE}")
    
    # Log some model information and data information
    flop, param = get_model_infos(base_model, input_shape)
    logger.log("model ====>>>>:\n{:}".format(base_model))
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    logger.log("-" * 50)
    logger.log("train_data : {:}".format(train_data))
    logger.log("valid_data : {:}".format(val_data))
    optimizer, scheduler, criterion = get_optim_scheduler(
        base_model.parameters(), optim_config
    )
    logger.log("optimizer  : {:}".format(optimizer))
    logger.log("scheduler  : {:}".format(scheduler))
    logger.log("criterion  : {:}".format(criterion))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )

    # Prepare network for training
    network, criterion = torch.nn.DataParallel(base_model).cuda(), criterion.cuda()
    logger.log("Using CUDA: ")

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_infox = torch.load(last_info)
        start_epoch = last_infox["epoch"] + 1
        last_checkpoint_path = last_infox["last_checkpoint"]
        if not last_checkpoint_path.exists():
            logger.log(
                "Does not find {:}, try another path".format(last_checkpoint_path)
            )
            last_checkpoint_path = (
                last_info.parent
                / last_checkpoint_path.parent.name
                / last_checkpoint_path.name
            )
        checkpoint = torch.load(last_checkpoint_path)
        base_model.load_state_dict(checkpoint["base-model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        valid_accuracies = checkpoint["valid_accuracies"]
        max_bytes = checkpoint["max_bytes"]
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    elif RESUME_PATH is not None:
        assert Path(RESUME_PATH).exists(), "Can not find the resume file : {:}".format(
            RESUME_PATH
        )
        checkpoint = torch.load(RESUME_PATH)
        start_epoch = checkpoint["epoch"] + 1
        base_model.load_state_dict(checkpoint["base-model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        valid_accuracies = checkpoint["valid_accuracies"]
        max_bytes = checkpoint["max_bytes"]
        logger.log(
            "=> loading checkpoint from '{:}' start with {:}-th epoch.".format(
                RESUME_PATH, start_epoch
            )
        )
    elif INIT_MODEL_PATH is not None:
        assert Path(
            INIT_MODEL_PATH
        ).exists(), "Can not find the initialization file : {:}".format(INIT_MODEL_PATH)
        checkpoint = torch.load(INIT_MODEL_PATH)
        base_model.load_state_dict(checkpoint["base-model"])
        start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}
        logger.log("=> initialize the model from {:}".format(INIT_MODEL_PATH))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}

    train_func, valid_func = get_procedures(PROCEDURE)

    total_epoch = optim_config.epochs + optim_config.warmup
    # Main Training and Evaluation Loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(start_epoch, total_epoch):
        scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (total_epoch - epoch), True)
        )
        epoch_str = "epoch={:03d}/{:03d}".format(epoch, total_epoch)
        LRs = scheduler.get_lr()
        find_best = False
        # set-up drop-out ratio
        if hasattr(base_model, "update_drop_path"):
            base_model.update_drop_path(
                model_config.drop_path_prob * epoch / total_epoch
            )
        logger.log(
            "\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], scheduler={:}".format(
                time_string(), epoch_str, need_time, min(LRs), max(LRs), scheduler
            )
        )

        # train for one epoch
        train_loss, train_acc1 = train_func(
            train_loader,
            network,
            criterion,
            scheduler,
            optimizer,
            optim_config,
            epoch_str,
            PRINT_FREQUENCY,
            logger,
        )
        # log the results
        logger.log(
            "***{:s}*** TRAIN [{:}] loss = {:.6f}, accuracy-1 = {:.2f}".format(
                time_string(), epoch_str, train_loss, train_acc1
            )
        )

        # evaluate the performance
        if (epoch % EVAL_FREQUENCY == 0) or (epoch + 1 == total_epoch):
            logger.log("-" * 150)
            valid_loss, valid_acc1 = valid_func(
                valid_loader,
                network,
                criterion,
                optim_config,
                epoch_str,
                PRINT_FREQUENCY_EVAL,
                logger,
            )
            valid_accuracies[epoch] = valid_acc1
            logger.log(
                "***{:s}*** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, | Best-Valid-Acc@1={:.2f}, Error@1={:.2f}".format(
                    time_string(),
                    epoch_str,
                    valid_loss,
                    valid_acc1,
                    valid_accuracies["best"],
                    100 - valid_accuracies["best"],
                )
            )
            if valid_acc1 > valid_accuracies["best"]:
                valid_accuracies["best"] = valid_acc1
                find_best = True
                logger.log(
                    "Currently, the best validation accuracy found at {:03d}-epoch :: acc@1={:.2f}, error@1={:.2f}, save into {:}.".format(
                        epoch,
                        valid_acc1,
                        100 - valid_acc1,
                        model_best_path,
                    )
                )
            num_bytes = (
                torch.cuda.max_memory_cached(next(network.parameters()).device) * 1.0
            )
            logger.log(
                "[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]".format(
                    next(network.parameters()).device,
                    int(num_bytes),
                    num_bytes / 1e3,
                    num_bytes / 1e6,
                    num_bytes / 1e9,
                )
            )
            max_bytes[epoch] = num_bytes
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch,
                "args": "none",
                "max_bytes": deepcopy(max_bytes),
                "FLOP": flop,
                "PARAM": param,
                "valid_accuracies": deepcopy(valid_accuracies),
                "model-config": model_config._asdict(),
                "optim-config": optim_config._asdict(),
                "base-model": base_model.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_base_path,
            logger,
        )
        if find_best:
            copy_checkpoint(model_base_path, model_best_path, logger)
        last_info = save_checkpoint(
            {
                "epoch": epoch,
                "args": "none",
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 200)
    logger.log(
        "Finish training/validation in {:} with Max-GPU-Memory of {:.2f} MB, and save final checkpoint into {:}".format(
            convert_secs2time(epoch_time.sum, True),
            max(v for k, v in max_bytes.items()) / 1e6,
            logger.path("info"),
        )
    )
    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    main()
