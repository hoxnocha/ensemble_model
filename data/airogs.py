from pathlib import Path
from typing import Any, Tuple
import torch
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms as T
from random import sample
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import pytorch_lightning as pl

from sklearn.model_selection import KFold
from argparse import ArgumentParser



class AirogsDataModule(pl.LightningDataModule):
    def __init__(self, 
                 task,
                 image_folder_path, 
                 image_csv_path, 
                 
                 transform,
                 fold_num, # fold number
                 split_seed,
                 kfold,
                 

                 train_batch_size, 
                 test_batch_size, 
                 num_workers, 
                 pin_memory,
                 ) -> None:
        
        super().__init__()
        self.task = task
        self.image_folder_path = image_folder_path
        self.image_csv_path = image_csv_path
        self.transform = transform
        self.fold_num = fold_num
        self.split_seed = split_seed
        self.kfold = kfold

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Airogs")
        parser.add_argument("--task", type=str, default="classification")
        parser.add_argument("--image_folder_path", type=str, default="/work/scratch/tyang/yolov5results")
        parser.add_argument("--image_csv_path", type=str, default="/images/innorestvision/eye/airogs/train_labels.csv")
        parser.add_argument("--transform", type=T.Compose, default=T.Compose([
                                    T.ToPILImage(),
                                    
                                    T.Resize((384,384), interpolation=Image.BICUBIC),
                                    T.CenterCrop(380),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                ]))
        parser.add_argument("--kfold", type=int, default=0)
        parser.add_argument("--fold_num", type=int, default=0)
        parser.add_argument("--split_seed", type=int, default=12345)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--test_batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--pin_memory", type=bool, default=True)
        return parent_parser




    def setup(self, kfold, fold_num, split_seed, stage: [str] = None) -> None:
        """Setup the data module.

        Args:
            stage (Optional[str], optional): Stage of the data module setup. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = self.get_train_val_datasets(kfold=kfold, fold_num=fold_num, split_seed=split_seed)
        if stage == "test" or stage is None:
            self.test_dataset = self.get_test_dataset()

    def get_train_val_datasets(self, kfold, fold_num, split_seed):
        """Get the train and validation datasets.

        Returns:
            Tuple[Dataset, Dataset]: Train and validation datasets.
        """
        dataset = AirogsDataset(task=self.task, 
                                image_folder_path=self.image_folder_path,
                                image_csv_path=self.image_csv_path, 
                                transform=self.transform)
        
        if kfold != 0:
            kf = KFold(n_splits=kfold, shuffle=True, random_state=split_seed)
            all_splits = [k for k in kf.split(dataset)]
            train_idx, val_idx = all_splits[fold_num]
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
            train_data, val_data = dataset[train_idx], dataset[val_idx]

        else:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

        return train_data, val_data
    
    def get_test_dataset(self):
        """Get the test dataset.

        Returns:
            Dataset: Test dataset.
        """
        dataset = AirogsDataset(task=self.task, 
                                image_folder_path=self.image_folder_path,
                                image_csv_path=self.image_csv_path, 
                                transform=self.transform)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        return  test_dataset
    

        
       

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get the train dataloader.

        Returns:
            TRAIN_DATALOADERS: Train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            
           
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get the validation dataloader.

        Returns:
            EVAL_DATALOADERS: Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get the test dataloader.

        Returns:
            EVAL_DATALOADERS: Test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            
        )  

class AirogsDataset(Dataset):


    def __init__(self, 
                 task, 
                 image_folder_path,
                 image_csv_path, 
                 transform, ) :
        super().__init__()
        self.task = task
        self.image_folder_paths = {f"cat{i}_path": image_folder_path + f"cat{i}/crops/od" for i in range(6)}
        self.image_csv_path = image_csv_path


        dfs = [self.process_category(cat_path) for cat_path in self.image_folder_paths.values()]
        self.df = pd.concat(dfs)
        label_mapping = {"NRG": 0, "RG":1 }
        self.df['class'] = self.df['class'].map(label_mapping)
    
            
        self.transform = transform    
        
    
    def process_category(self, cat_path):
            files = glob.glob1(cat_path, "*.jpg")
            files = [os.path.basename(file)[:-4] for file in files]
            df_cat = self.df[self.df['challenge_id'].isin(files)]
            df_cat["challenge_id"] = f"{cat_path}/" + df_cat["challenge_id"] + ".jpg"
            return df_cat
    

    def hsv_equalize(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        v_eq = cv2.equalizeHist(v)
        image = cv2.merge((h, s, v_eq))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image
            
        

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.df)
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:

        image_path = self.df.iloc[index]['challenge_id']

        image = cv2.imread(str(image_path))
        image = self.hsv_equalize(image)
        image = self.transform(image)
        label = self.df.iloc[index]['class']
        label = torch.tensor(label, dtype=torch.float32)
        

        return image, label
    
    

