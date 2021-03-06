"""
Utilities for downloading and processing the UCI Energy Efficiency Dataset
2012.
"""

from pathlib import Path
from typing import Tuple

import pandas
import numpy as np
import torch

TRAIN_FILENAME = 'energy_efficiency_2012_train.csv'
TEST_FILENAME = 'energy_efficiency_2012_test.csv'
DATA_URL = ('https://archive.ics.uci.edu/'
            'ml/machine-learning-databases/00242/ENB2012_data.xlsx')


def download_data(url: str, train_path: str, test_path: str):
    """
    Download the UCI Energy Efficiency Dataset 2012 from the provided url,
    split it into train and test, and save the splits as CSV files.
    Args:
        url: URL from which to download the dataset.
        train_path: Path at which to store the training data.
        test_path: Path at which to store the test data.
    """
    print(f"Downloading dataset from {url}")
    data_frame = pandas.read_excel(url)

    # It looks like the first 48 rows of the dataset are intended to be a test
    # set.
    # We can't use a typical shuffled train-test split because each row isn't
    # independent from its neighbours.
    # This is why the model we have doesn't *appear* to do as well as other
    # models on the internet, whereas infact it will generalize better.
    test = data_frame.iloc[:48]
    train = data_frame.iloc[48:]

    train.to_csv(train_path)
    test.to_csv(test_path)


class EnergyEfficiencyDataset(torch.utils.data.Dataset):
    """
    UCI Energy Efficiency Dataset 2012
    """

    def __init__(self,
                 data_dir: str,
                 train: bool = True,
                 download: bool = True,
                 transform: bool = None):
        """
        Args:
            data_dir: Directory in which the data is stored.
            train: Whether to use the training split or the test split.
            download: Whether or not to download the data if it doesn't exist
                in the data directory.
            transform: Optional transform to be applied on a sample.
        """
        if train:
            filepath = Path(data_dir) / TRAIN_FILENAME
        else:
            filepath = Path(data_dir) / TEST_FILENAME

        if not filepath.is_file():
            if not download:
                raise RuntimeError("Data doesn't exist and isn't set to be "
                                   "downloaded, set download=True to download "
                                   "it")
            download_data(DATA_URL,
                          Path(data_dir) / TRAIN_FILENAME,
                          Path(data_dir) / TEST_FILENAME)

        self.dataframe = pandas.read_csv(filepath)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.dataframe.iloc[idx, 1:9].to_numpy(dtype=np.float32)
        targets = self.dataframe.iloc[idx, 9:11].to_numpy(dtype=np.float32)

        if self.transform:
            features = self.transform(features)

        return features, targets
