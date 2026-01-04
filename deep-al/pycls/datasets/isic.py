from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from typing import Any
import pandas as pd

class ISIC(Dataset):
    def __init__(self, root: str, train, transform=None, test_transform=None, only_features=False, **kwargs: Any) -> None:
        self.train = train
        mode = 'train' if self.train else 'test' 
        self.root = root
        self.test_transform = test_transform
        self.transform = transform

        self.img_dir = os.path.join(self.root, "ISIC_2019_Training_Input_preprocessed")
        self.train_and_split = os.path.join(self.root, "train_test_split")
        df = pd.read_csv(self.train_and_split)
        df_train = df[df['fold'] == mode]
        self.data_list = list(df_train['image'])
        self.targets = list(df_train['target'])
        self.classes = ['melanoma', 'melanocytic nevus', 'basal cell carcinoma', 'actinic keratosis', 'benign keratosis', 'dermatofibroma', 'vascular lesion', 'squamous cell carcinoma']

        self.only_features = only_features
        if self.only_features:
            if self.train:
                self.features = np.load('../../results/isic/pretext/features_seed2026.npy')
            else:
                self.features = np.load('../../results/isic/pretext/test_features_seed2026.npy')
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path = self.img_dir +'/'+ self.data_list[index] +'.jpg'
        target = self.targets[index]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')

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
        
        