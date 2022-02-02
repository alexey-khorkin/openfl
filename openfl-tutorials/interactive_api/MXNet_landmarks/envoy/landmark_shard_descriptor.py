# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Landmarks Shard Descriptor."""

import logging
import zipfile
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class LandmarkShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        X, y = self.download_data()
        self.X = X[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                  random_state=42, shuffle=True)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            X_train, X_test, y_train, y_test
        
        self.dataset_mode = 'train'

    def set_dataset_type(self, mode='train'):
        """Select training or testing mode."""
        self.dataset_mode = mode

    def get_train_size(self):
        """Return train dataset size."""
        return len(self.X_train)

    def get_test_size(self):
        """Return test dataset size."""
        return len(self.X_test)

    @staticmethod
    def process_data(path_to_csv_file):
        data_df = pd.read_csv(path_to_csv_file)
        data_df.fillna(method = 'ffill', inplace = True)
        labels = data_df.drop('Image', axis = 1)
        imag, keypoints = [], []
        for i in range(data_df.shape[0]):
            img = data_df['Image'][i].split(' ')
            img = ['0' if x == '' else x for x in img]
            imag.append(img)
            y = labels.iloc[i, :]
            keypoints.append(y)

        X = np.array(imag,dtype = 'float').reshape(-1, 96, 96)
        y = np.array(keypoints, dtype = 'float')
        
        return X, y

    def download_data(self):
        """Download prepared dataset."""
        local_file_path = 'facial-keypoints-detection.zip'
        target_path = 'facial-keypoints-detection'

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        
        # unpack training.zip
        os.path.join(target_path, 'training.zip')
        with zipfile.ZipFile(os.path.join(target_path, 'training.zip'), 'r') as zip_ref:
            zip_ref.extractall(target_path)

        X, y = self.process_data(os.path.join(target_path, 'training.csv'))
        return X, y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        if self.dataset_mode == 'train':
            return self.X_train[index], self.y_train[index]
        else:
            return self.X_test[index], self.y_test[index]
    
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['96', '96']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Landmark dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def __len__(self):
        """Return the len of the dataset."""
        if self.dataset_mode is None:
            return 0

        if self.dataset_mode == 'train':
            return len(self.X_train)
        else:
            return len(self.X_test)


if __name__ == "__main__":
    shadr = LandmarkShardDescriptor('1, 2')
