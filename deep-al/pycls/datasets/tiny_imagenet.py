import os
import numpy as np
from PIL import Image

import torchvision.datasets as datasets
import sys
from typing import Any
class TinyImageNet(datasets.VisionDataset):
    """`Tiny ImageNet Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        samples (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root: str, split: str = 'train', transform=None, test_transform=None, only_features=False, **kwargs: Any) -> None:
        self.train = (split == 'train')
        self.root = root
        self.test_transform = test_transform
        self.train_dir = os.path.join(self.root, "train")
        self.val_dir = os.path.join(self.root, "val")

        assert self.check_root(), "Something is wrong with the Tiny ImageNet dataset path. Download the official dataset zip from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it inside {}.".format(self.root)
        # wnid_to_classes = self.load_wnid_to_classes()
        self.transform = transform
        if (self.train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()
        
        self._make_dataset(self.train)

        words_file = os.path.join(self.root, "words.txt")
        wnids_file = os.path.join(self.root, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        self.only_features = only_features
        if self.train:
            self.features = np.load('../../results/tiny-imagenet/pretext/features_seed2026.npy')
        else:
            self.features = np.load('../../results/tiny-imagenet/pretext/test_features_seed2026.npy')

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        self.classes = classes
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.val_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.val_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        self.classes = classes
        num_images = 0
        for root, dirs, files in os.walk(self.val_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        self.targets = []
        if Train:
            img_root_dir = self.train_dir
        else:
            img_root_dir = self.val_dir
        list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        with open(path, 'rb') as f:
                            sample = Image.open(path)
                            sample = sample.convert('RGB')
                            self.images.append(sample)
                            self.targets.append(self.class_to_tgt_idx[tgt])


    def check_root(self):
        tinyim_set = ['words.txt', 'wnids.txt', 'train', 'val', 'test']
        for x in os.scandir(self.root):
            if x.name not in tinyim_set:
                return False
        return True
    def __len__(self):
        return self.len_dataset
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.images[index], self.targets[index]

        if self.only_features:
            sample = self.features[index]
        else:
            if self.train:
                if self.transform is not None:
                    sample = self.transform(sample)
            else:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)

        return sample, index, target