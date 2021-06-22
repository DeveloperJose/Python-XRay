import os
import json
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from nni.utils import merge_parameter
from torch.utils.data import DataLoader

from xray_dataset import XrayImageDataset

logger = logging.getLogger('xray_AutoML')


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(186050, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def compute_loss(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def prepare_batch(self, x, y, device):
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(1).to(torch.float)
        return x, y


def train(args, model, device, train_loader, optimizer):
    total_loss = 0
    correct_predictions = 0

    model.train()
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        if (args['batch_num'] is not None) and batch_idx >= args['batch_num']:
            break

        x_train, y_train = model.prepare_batch(x_train, y_train, device)
        
        optimizer.zero_grad()
        output = model(x_train)
        loss = model.compute_loss(output, y_train)
        loss.backward()
        optimizer.step()

        # Update epoch metrics
        y_pred = output.round()
        total_loss += loss
        correct_predictions += y_pred.eq(y_train).sum().item()

    # Compute final metrics
    average_train_loss = total_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * correct_predictions / len(train_loader.dataset)
    return average_train_loss, train_accuracy

def validate(args, model, device, val_loader):
    total_loss = 0
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = model.prepare_batch(x_val, y_val, device)

            output = model(x_val)
            y_pred = output.round()

            # Update validation metrics
            total_loss += model.compute_loss(output, y_val).item()
            correct_predictions += y_pred.eq(y_val).sum().item()

    # Compute final validation metrics
    average_val_loss = total_loss / len(val_loader.dataset)
    val_accuracy = 100. * correct_predictions / len(val_loader.dataset)
    return average_val_loss, val_accuracy

def test(args, model, device, test_loader):
    model.eval()
    indices = []
    predictions = []

    logger.info("[Test] Processing test data")
    with torch.no_grad():
        for x_test, image_id in test_loader:
            x_test = x_test.to(device)
            output = model(x_test)

            # From the sigmoid output, round and then convert from float to int
            # Then convert from torch tensor to a numpy array and then to a Python list
            y_pred = output.round().to(torch.int).numpy().tolist()

            indices.extend(image_id)
            predictions.extend(y_pred)

    df = pd.DataFrame({'ImageId': indices, 'Label': predictions})
    df.to_csv(args['output_path'], index=False)
    logger.info(f"[Test] Saved submission file to {args['output_path']}")

def save_checkpoint(args, epoch, model, optimizer, loss, accuracy):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
            }, args['checkpoint_path'])

def load_checkpoint(args, model, optimizer):
    checkpoint = torch.load(args['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def main(args):
    # CUDA preparation
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    logger.info(f"Using CUDA: {use_cuda}")

    # Dataset preparation
    data_dir = args['data_dir']
    debugging = args['debugging']
    train_data, val_data, test_data, x_shape, num_classes = XrayImageDataset.get_datasets(data_dir, debugging)

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=True, **kwargs)

    # Model and optimizer preparation
    hidden_size = args['hidden_size']
    model = Net(hidden_size=hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    logger.info('Model=\n' + str(model))
    logger.info('Optimizer=\n' + str(optimizer))

    # Checkpoint preparation
    best_val_accuracy = 0

    for epoch in range(1, args['epochs'] + 1):
        average_train_loss, train_accuracy = train(args, model, device, train_loader, optimizer)
        logger.info(f'[Training] Epoch={epoch} | Average Loss: {average_train_loss:.4f} | Accuracy: {train_accuracy:.0f}%')

        average_val_loss, val_accuracy = validate(args, model, device, val_loader)
        logger.info(f'[Validation] Average Loss: {average_val_loss:.4f} | Accuracy: {val_accuracy:.0f}')

        # Report intermediate result to NNI
        nni.report_intermediate_result(val_accuracy)
        logger.debug(f'Sent intermediate result to NNI ({val_accuracy})')

        # Check if this model is our best one, if so, save the checkpoint
        if val_accuracy > best_val_accuracy:
            logger.info(f"This model is better than our previous best one ({val_accuracy}>{best_val_accuracy}), attempting to save checkpoint")
            best_val_accuracy = val_accuracy
            save_checkpoint(args, epoch, model, optimizer, average_val_loss, val_accuracy)
            logger.info(f"Saved model to {args['checkpoint_path']}")

    # Report final result for NNI
    nni.report_final_result(val_accuracy)
    logger.debug(f'Sent final result to NNI ({val_accuracy})')

    # Generate the submission file with the test set
    test(args, model, device, test_loader)
    logger.debug('Finished creating submission file')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training file to use NNI on the XRay Dataset')
    
    # Special parameters
    parser.add_argument("--config_path", type=str, default=None, help='JSON file with the parameters not included in the search space')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument("--debugging", action='store_true', default=False, help='Sets the dataset to a small size for debugging purposes')

    # Base parameters
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset base directory")
    parser.add_argument('--batch_size', type=int, default=None, help='Input batch size for training')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=None, help='How many batches to wait before logging training status')
    parser.add_argument('--output_path', type=str, default=None, help='Filepath for test set submission CSV output file')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Filepath for best model checkpoint file')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # Get the default parameters
        params = vars(get_params())

        # Load the JSON parameter file and override those parameters which are set there
        with open(params['config_path']) as config_file:
            config_json = json.load(config_file)
            params = merge_parameter(params, config_json)

        # Get searched parameters from NNI tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)

        # Take the base parameters and override those which are changed by the tuner
        params = merge_parameter(params, tuner_params)

        logger.info("Params=\n" + str(params))
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
