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

class AirogsDataset(Dataset):


    def __init__(self, 
                 task, 
                 image_folder_path,
                 image_csv_path, 
                 transform 
            ) -> None:
        super().__init__()
        self.task = task
        self.cat0_path = image_folder_path + "/cat0/crops/od"
        self.cat1_path = image_folder_path + "/cat1/crops/od"
        self.cat2_path = image_folder_path + "/cat2/crops/od"
        self.cat3_path = image_folder_path + "/cat3/crops/od"
        self.cat4_path = image_folder_path + "/cat4/crops/od"
        self.cat5_path = image_folder_path + "/cat5/crops/od"
        
        self.df = pd.read_csv(image_csv_path)
        files0 = glob.glob1(self.cat0_path, '*.jpg')
        files0 = [os.path.basename(file)[:-4] for file in files0]
        self.df0 = self.df[self.df['challenge_id'].isin(files0)]
        self.df0["challenge_id"] = f"{self.cat0_path}" + "/" + self.df['challenge_id'] + '.jpg'
        files1 = glob.glob1(self.cat1_path, '*.jpg')
        files1 = [os.path.basename(file)[:-4] for file in files1]
        self.df1 = self.df[self.df['challenge_id'].isin(files1)] 
        self.df1["challenge_id"] = f"{self.cat1_path}" + "/" + self.df['challenge_id'] + '.jpg'
        files2 = glob.glob1(self.cat2_path, '*.jpg')
        files2 = [os.path.basename(file)[:-4] for file in files2]
        self.df2 = self.df[self.df['challenge_id'].isin(files2)]
        self.df2["challenge_id"] = f"{self.cat2_path}" + "/" + self.df['challenge_id'] + '.jpg'
        files3 = glob.glob1(self.cat3_path, '*.jpg')
        files3 = [os.path.basename(file)[:-4] for file in files3]
        self.df3 = self.df[self.df['challenge_id'].isin(files3)]
        self.df3["challenge_id"] = f"{self.cat3_path}" + "/" + self.df['challenge_id'] + '.jpg'
        files4 = glob.glob1(self.cat4_path, '*.jpg')
        files4 = [os.path.basename(file)[:-4] for file in files4]
        self.df4 = self.df[self.df['challenge_id'].isin(files4)]
        self.df4["challenge_id"] = f"{self.cat4_path}" + "/" + self.df['challenge_id'] + '.jpg'
        files5 = glob.glob1(self.cat5_path, '*.jpg')
        files5 = [os.path.basename(file)[:-4] for file in files5]
        self.df5 = self.df[self.df['challenge_id'].isin(files5)]
        self.df5["challenge_id"] = f"{self.cat5_path}" + "/" + self.df['challenge_id'] + '.jpg'
       
        self.df = pd.concat([self.df0, self.df1, self.df2, self.df3, self.df4, self.df5])
        
        label_mapping = {"NRG": 0, "RG":1 }
        self.df['class'] = self.df['class'].map(label_mapping)
        
        self.transform = transform    
        
        

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.df)
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:

        image_path = self.df.iloc[index]['challenge_id']

        image = cv2.imread(str(image_path))
        image = cv2.equalizeHist(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = self.df.iloc[index]['class']
        label = torch.tensor(label)
        

        return image, label
    
    
        
