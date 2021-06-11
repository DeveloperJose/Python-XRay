#####################################################
# Author: Jose G. Perez
# Modified from AutoDL-Projects/exps/basic/basic-main.py (by Xuanyi Dong [GitHub D-X-Y])
#####################################################
import os, sys, time, torch, random, argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy

from xautodl.config_utils import load_config, dict2config
from xautodl.procedures import get_procedures, get_optim_scheduler
from xautodl.datasets import get_datasets
from xautodl.models import obtain_model
from xautodl.utils import get_model_infos
from xautodl.log_utils import PrintLogger, time_string

import pandas as pd
import numpy as np
from tqdm import tqdm

from xray_dataset import XrayImageDataset

def main():
    checkpoint_filepath = './output/simple_model/checkpoint/seed-42-best.pth'
    output_filepath = './output/simple_model/submission.csv'

    assert os.path.isfile(checkpoint_filepath), f'Checkpoint filepath ({checkpoint_filepath}) is not a file or does not exist'

    checkpoint = torch.load(checkpoint_filepath)

    train_data, val_data, test_data, x_shape, num_classes = XrayImageDataset.get_datasets()

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger = PrintLogger()
    model_config = dict2config(checkpoint["model-config"], logger)
    base_model = obtain_model(model_config)
    flop, param = get_model_infos(base_model, x_shape)
    logger.log("model ====>>>>:\n{:}".format(base_model))
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    logger.log("-" * 50)

    optim_config = dict2config(checkpoint["optim-config"], logger)
    _, _, criterion = get_optim_scheduler(base_model.parameters(), optim_config)
    logger.log("criterion  : {:}".format(criterion))
    base_model.load_state_dict(checkpoint["base-model"])

    network = torch.nn.DataParallel(base_model).cuda()

    logger.log("Evaluating test set")
    network.eval()
    indices = []
    predictions = []

    with torch.no_grad():
        with tqdm(test_loader, unit='batch') as t_bar:
            for i, (x, image_id) in enumerate(t_bar):
                t_bar.set_description(f"Batch {i}")

                features, logits = network(x)
                y_pred = torch.round(torch.max(features, dim=1).values)
                y_pred_np = y_pred.cpu().numpy().astype(np.uint8)

                indices.extend(image_id)
                predictions.extend(y_pred_np.tolist())

    df = pd.DataFrame({'ImageId': indices, 'Label': predictions})
    df.to_csv(output_filepath, index=False)

    logger.close()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "torch.cuda is not available"
    main()
