import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tifffile
from config import *
from torch.utils.data import Dataset, DataLoader
from .augmentation import augmentation
import torch


class CustomImageDataset(Dataset):
    
    def __init__(self,
                 path_images, 
                 dataframe,
                 augmentation=True):
        
        self.path_images = path_images
        self.dataframe = dataframe
        self.augmentation = augmentation

    def __getitem__(self, idx):

        filename = self.dataframe.iloc[idx]['filename']
        """read image"""
        image = tifffile.imread(os.path.join(self.path_images, filename))/255
        if self.augmentation:
            image = augmentation(image)
        """ Generate output grids by comparing modified baseline seg with ground truth"""
        input_= np.transpose(image, (2,0,1)).astype(np.float32)
        output_ = input_

        return (torch.tensor(input_), torch.tensor(output_))
    
    def __len__(self):
        
        return self.dataframe.shape[0]


dataset_train = CustomImageDataset(path_images = path_images,
                                   dataframe = df_train,
                                   augmentation=True)


dataset_test = CustomImageDataset(path_images = path_images,
                                  dataframe = df_test,
                                  augmentation=False)

dataset_val = CustomImageDataset(path_images = path_images,
                                 dataframe = df_val,
                                 augmentation=False)


loader_train = DataLoader(
    batch_size = 32,
    dataset = dataset_train,
    num_workers = 16,
    shuffle = True)

loader_val = DataLoader(
    batch_size = 32,
    dataset = dataset_val,
    num_workers = 16,
    shuffle = False)

loader_test = DataLoader(
    batch_size = 32,
    dataset = dataset_test,
    num_workers = 16,
    shuffle = False)