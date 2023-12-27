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
from airogs_basemodel.data.airogs_label import LABEL_DICT

class AirogsDataset(Dataset):


    def __init__(self, task, image_folder_path, csv_file_path, ) -> None:
        super().__init__()
        self.task = task
        self.image_folder_path = image_folder_path
        self.df = pd.read_csv(csv_file_path)
        files = glob.glob1(self.image_folder_path, '*.jpg')
        files = [os.path.basename(file)[:-4] for file in files]
        self.df = self.df[self.df['challenge_id'].isin(files)]
        label_mapping = {"NRG": 0, "RG":1 }
        self.df['class'] = self.df['class'].map(label_mapping)
        self.transform =  T.Compose([
            T.ToPILImage(),
            T.Resize(384, interpolation=Image.BICUBIC),
            T.CenterCrop(380),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            
        
        

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.df)
    
    def __getitem__(self, index: int) -> tuple[Any, Any]:

        #get image path
        image_path = f"{self.image_folder_path}" + "/" + self.df.iloc[index]['challenge_id'] + '.jpg'
        #image_path = Path(image_path)
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        v_equlized = cv2.equalizeHist(v)
        #import ipdb; ipdb.set_trace()
        image = cv2.merge((h, s, v_equlized))
        image = self.transform(image)
        label = self.df.iloc[index]['class']
        label = torch.tensor(label)
        

        return image, label
    
    
        
