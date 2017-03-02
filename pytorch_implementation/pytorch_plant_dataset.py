"""
Dataset implementation for our
"""

import glob
import logging
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import dataset

logging.basicConfig(level=logging.DEBUG)


class PlantDataset(dataset.Dataset):
    """
    We extend the PyTorch dataset class, zie: http://pytorch.org/docs/data.html
    """

    def __init__(self, root, data_size, transform=None, target_transform=None, image_size=32, train=True):
        """
        :param root: Path to datasets, train datasets are prefixed with 'train_' and test datasets are prefixed with
        'test_'. There will be search recursively so just put these files somewhere in the root directory and they
        will be found.
        :param data_size: data size per file/class
        :param transform: transform object
        """
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.data_size = data_size

        self.total_data_size = 0

        if train:
            files = glob.glob(root + '/**/train_*', recursive=True)
        else:
            files = glob.glob(root + '/**/test_*', recursive=True)

        logging.info('Found the following training files: {}'.format(files))

        self.train_data = []
        self.train_labels = []
        for file in files:
            logging.info('Opening pickled train data {}'.format(file))
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
                self.train_labels.extend(data['labels'])
                self.train_data.append(data['data'])
            self.total_data_size += data_size

        print("self.train_data:")
        print(self.train_data)
        print("shape: ")
        print(len(self.train_data))
        
        self.train_data = np.concatenate(self.train_data)
        self.train_data = self.train_data.reshape((data_size, 3, image_size, image_size))

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data_size
