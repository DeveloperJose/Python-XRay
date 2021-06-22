import os
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

def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_dir = args['data_dir']
    train_data, val_data, test_data, x_shape, num_classes = XrayImageDataset.get_datasets(data_dir)

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=True, **kwargs)

    hidden_size = args['hidden_size']
    model = Net(hidden_size=hidden_size).to(device)
    logger.info('Model=\n' + str(model))
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        average_train_loss, train_accuracy = train(args, model, device, train_loader, optimizer)
        logger.info(f'[Training] Epoch={epoch} | Average Loss: {average_train_loss:.4f} | Accuracy: {train_accuracy:.0f}%')

        average_val_loss, val_accuracy = validate(args, model, device, val_loader)
        logger.info(f'[Validation] Average Loss: {average_val_loss:.4f} | Accuracy: {val_accuracy:.0f}')

        # Report intermediate result to NNI
        nni.report_intermediate_result(val_accuracy)
        logger.debug(f'Sent intermediate result to NNI ({val_accuracy})')

    # Report final result for NNI
    nni.report_final_result(val_accuracy)
    logger.debug(f'Sent final result to NNI ({val_accuracy})')

    # Generate the submission file with the test set
    test(args, model, device, test_loader)
    logger.debug('Finished creating submission file')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='AutoML XRay Script')
    parser.add_argument("--data_dir", type=str, default='/data/datasets/xray-dataset/v2/', help="data directory")
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',help='input batch size for training')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=256, metavar='N', help='hidden layer size')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--output_path', type=str, default='/data/jperez/git-python-xray/nni/submission.csv', help='where to store the test set submission file')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
