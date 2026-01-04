import torch
import torchvision
from PIL import Image
import numpy as np
import pycls.datasets.utils as ds_utils


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform, download=True, only_features= False):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.train = train
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR10", train=train, normalized=False)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.train:
                if self.transform is not None:
                    img = self.transform(img)
            else:
                if self.test_transform is not None:
                    img = self.test_transform(img)


        return img, index, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform, download=True, only_features= False):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.train = train
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR100", train=train, normalized=False)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.train:
                if self.transform is not None:
                    img = self.transform(img)
            else:
                if self.test_transform is not None:
                    img = self.test_transform(img)

        return img, index,  target
