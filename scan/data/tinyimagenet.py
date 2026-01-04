"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
from PIL import Image
from torchvision.datasets.utils import check_integrity
from torchvision import datasets
from typing import Any
import sys
from utils.mypath import MyPath

class TinyImageNet(datasets.VisionDataset):
    def __init__(self, root = MyPath.db_root_dir('tiny-imagenet'), split='train', transform=None):
        self.Train = (split == 'train')
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

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
                        item = (path, self.class_to_tgt_idx[tgt])
                        self.images.append(item)
                        self.targets.append(self.class_to_tgt_idx[tgt])

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        out = {'image': sample, 'target': tgt, 'meta': {'im_size': 64, 'index': idx, 'class_name': tgt}}
        return out

def unpickle_object(path):
    with open(path, 'rb+') as file_pi:
        res = pickle.load(file_pi)
    return res


# class TinyImageNet(datasets.VisionDataset):
#     """`Tiny ImageNet Classification Dataset.

#     Args:
#         root (string): Root directory of the ImageNet Dataset.
#         split (string, optional): The dataset split, supports ``train``, or ``val``.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.

#      Attributes:
#         classes (list): List of the class name tuples.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         wnids (list): List of the WordNet IDs.
#         wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
#         samples (list): List of (image path, class_index) tuples
#         targets (list): The class_index value for each image in the dataset
#     """

#     def __init__(self, root: str, split: str = 'train', transform=None, **kwargs: Any) -> None:
#         self.root = root
#         if split == 'train+unlabeled':
#             split = 'train'
#         self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val"))

#         if self.split == 'train':
#             self.images, self.targets, self.cls_to_id = unpickle_object('../../../daphna/data/tiny_imagenet/tiny-imagenet-200/train.pkl')
#         elif self.split == 'val':
#             self.images, self.targets, self.cls_to_id = unpickle_object('../../../daphna/data/tiny_imagenet/tiny-imagenet-200/val.pkl')
#         else:
#             raise NotImplementedError('unknown split')
#         self.targets = self.targets.astype(int)
#         self.classes = list(self.cls_to_id.keys())
#         super(TinyImageNet, self).__init__(root, **kwargs)
#         self.transform = transform

#     # Split folder is used for the 'super' call. Since val directory is not structured like the train,
#     # we simply use train's structure to get all classes and other stuff
#     @property
#     def split_folder(self) -> str:
#         return os.path.join(self.root, 'train')

#     def __getitem__(self, index: int):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         sample = Image.fromarray(self.images[index])
#         target = int(self.targets[index])

#         if self.transform is not None:
#             sample = self.transform(sample)

#         out = {'image': sample, 'target': target, 'meta': {'im_size': 64, 'index': index, 'class_name': target}}
#         return out

#     def __len__(self):
#         return len(self.targets)
