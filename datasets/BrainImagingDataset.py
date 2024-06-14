import nibabel as nib
import sys
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np
import torch as torch

from torchvision import transforms

from torchvision import transforms, utils
from torchvision.transforms import functional as tf
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.functional import auroc, accuracy

from  concurrent.futures import ThreadPoolExecutor, as_completed

sys.path
import warnings
warnings.simplefilter('ignore')


class BrainImgDataset():
    def __init__(self,root_dir:str, modality_path:str , table_dir,split = 'train') -> None:
        self.root_dir = root_dir
        self.table_dir = table_dir
        self.all_folders, self.all_labels = self.load_sex_table()
        self.modality_path =  modality_path
        self.existing_folders = self.check_paths_and_get_folders(self.all_folders)

        self.len_dataset = len(self.existing_folders)
        self.split = split
        self.index_split = (int(0.7*self.len_dataset), int(0.73*self.len_dataset)) 

        # find list of eid's of patients for train val and test set
        assert self.split in ['train','val','test']
        if self.split == 'train':
            self.folders = self.existing_folders[:self.index_split[0]]

        elif self.split == 'val':
            self.folders = self.existing_folders[self.index_split[0]:self.index_split[1]]

        elif self.split == 'test':
            self.folders = self.existing_folders[self.index_split[1]:]
            
        else:
            print('incorrect split variable')
        #load sex labels 
        self.labels = self.all_labels[self.all_labels.index.isin(self.folders)]['Sex'].to_list()
        
        #Load all the images

        self.imgs = self.load_images()
        #augmentations
        self.augmentation  = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(degrees=20),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
       transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1,2.0)),
                # transforms.RandomResizedCrop(size=self.size, scale=(0.2,1), ratio=(0.75, 1.2)),
                # transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
                transforms.RandomAffine(degrees=10,scale=(0.8, 1.2),shear=10),
            ],p=0.5),

             # transforms.Normalize(mean=0.5,std=0.5)
            ])

    
    def check_paths_and_get_folders(self,foldernames):
        def check_path(foldername):
            return foldername if os.path.exists(os.path.join(self.root_dir, str(foldername), self.modality_path)) else None
        existing_folders = []
    
        # Use ThreadPoolExecutor to parallelize the path checks
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a future for each path check
            futures = {executor.submit(check_path, foldername): foldername for foldername in foldernames}
        
        # As each future completes, process the result
        for future in as_completed(futures):
            result = future.result()
            if result:
                existing_folders.append(result)
        return existing_folders
    

    
    def load_sex_table(self):
        path_demographics = self.table_dir + 'ukb_cardiac_image_subset_Primary demographics.csv'
        df_demo = pd.read_csv(path_demographics)
        df_demo = df_demo.drop(df_demo.index[0])
        df_demo = df_demo.set_index('eid')
        df_demo['Sex'] = df_demo['Sex'].astype(str)
        df_demo = df_demo[['Sex']]

        num_unexpected_sex_val =len(df_demo.loc[~df_demo['Sex'].isin(['0', '1']), 'Sex'].unique())
        if  num_unexpected_sex_val != 0:
            print(f"Number of unexpected Sex sentries that aren't 1 or 0: {num_unexpected_sex_val}")
        return df_demo.index.to_list(), df_demo
    
    def find_min_size(self):
        heights = []
        widths = []
        depths = []
        
        # Load images and store their sizes
        for folder in self.folders:
            path = os.path.join(self.root_dir, str(folder), self.modality_path)
            img = nib.load(path)
            img_array_3d = img.get_fdata()
            img_array_slice = img_array_3d[:, :, 90]
            heights.append(img_array_slice.shape[0])
            widths.append(img_array_slice.shape[1])
            depths.append(img_array_slice.shape[2])
        min_height = min(heights)
        min_width = min(widths)
        min_depth = min(depths)
        return min_height, min_width, min_depth
    
    def load_images(self):
        imgs = []
        for folder in self.folders:
            path =os.path.join(self.root_dir,str(folder), self.modality_path)
            img = nib.load(path)
            img_array_3d = img.get_fdata()
            img_array_slice = img_array_3d[:,:,90]
            slice_tensor = torch.from_numpy(img_array_slice)
            #crop images to the same size
            # out = slice_tensor.unsqueeze(0).unsqueeze(0)
            # img_array_slice_cropped = transforms.functional.resize(out, self.size)
            imgs.append(slice_tensor.type('torch.DoubleTensor'))
        
            if not slice_tensor.shape == torch.Size([182, 218]):
                print(slice_tensor.shape)     
        return torch.stack(imgs)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self,idx):
        
        img_array_slice = self.imgs[idx]
        img_array_slice = img_array_slice.unsqueeze(0)
        label = self.labels[idx]
        img_view1=self.augmentation(img_array_slice)
        img_view2=self.augmentation(img_array_slice)   

        return img_view1, img_view2