import os

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class XrayImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, x_transform_func=None, debugging=False):
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.x_transform_func = x_transform_func
        self.debugging = debugging

        self.is_annotated = csv_file is not None
        print(f"[XRayImageDataset] Creating dataset v2, csv={csv_file}, debugging={self.debugging}")
        if self.is_annotated:
            self.df = pd.read_csv(csv_file, dtype={'ImageId': str, 'Label': int})
            self.df['ImagePath'] = self.df['ImageId'].apply(lambda file_id: os.path.join(self.img_dir, file_id + ".png"))

    def __len__(self):
        if self.debugging:
            return 10
        elif self.is_annotated:
            return len(self.df)
        else:
            return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.is_annotated:
            # Get the row from the dataframe given the index
            x = np.array(Image.open(self.df.iloc[idx]['ImagePath']))
            y = self.df.iloc[idx]['Label']
            if self.x_transform_func:
                x = self.x_transform_func(x)
            return x, torch.tensor(y)
        else:
            filename = os.listdir(self.img_dir)[idx]
            x = np.array(Image.open(os.path.join(self.img_dir, filename)))
            y = filename[:filename.rindex('.')]
            if self.x_transform_func:
                x = self.x_transform_func(x)
            return x, y

    @staticmethod
    def get_datasets(data_dir, debugging):
        dataset_dir = Path(data_dir)

        x_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

        train_data = XrayImageDataset(dataset_dir / 'train-set/', dataset_dir / 'train-labels.csv', x_transform_func=x_transform, debugging=debugging)
        val_data = XrayImageDataset(dataset_dir / 'validation-set/', dataset_dir / 'validation-labels.csv', x_transform_func=x_transform, debugging=debugging)
        test_data = XrayImageDataset(dataset_dir / 'test-set/', None, x_transform_func=x_transform, debugging=debugging)

        input_shape = (3, 256, 256)
        num_classes = 2
        return train_data, val_data, test_data, input_shape, num_classes
