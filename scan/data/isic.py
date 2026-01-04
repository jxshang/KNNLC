from utils.mypath import MyPath
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class ISIC(Dataset):
    def __init__(self, root = MyPath.db_root_dir('isic'), train=None, transform=None):
        self.Train =  train
        mode = 'train' if self.Train else 'test' 
        self.root_dir = root
        self.transform = transform
        self.img_dir = os.path.join(self.root_dir, "ISIC_2019_Training_Input_preprocessed")

        self.classes = ['melanoma', 'melanocytic nevus', 'basal cell carcinoma', 'actinic keratosis', 'benign keratosis', 'dermatofibroma', 'vascular lesion', 'squamous cell carcinoma']

        self.train_and_split = os.path.join(self.root_dir, "train_test_split")
        df = pd.read_csv(self.train_and_split)
        df_train = df[df['fold'] == mode]
        self.data_list = list(df_train['image'])
        self.target = list(df_train['target'])


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path = self.img_dir +'/'+ self.data_list[idx] +'.jpg'
        tgt = self.target[idx]
        class_name = self.classes[tgt]  
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
            img_size = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
        out = {'image': sample, 'target': tgt, 'meta': {'im_size': img_size, 'index': idx, 'class_name': class_name}}
        return out
        
