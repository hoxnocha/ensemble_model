from pandas import DataFrame
import pytorch_lightning as pl
from ensemble_model.data.dataset import AirogsDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import torch

class AirogsDataModule(pl.LightningDataModule):
    def __init__(self, 
                 #task, 
                 image_folder_path, 
                 rg_folder_path, 
                 image_csv_path, 
                 rg_csv_path, 
                 train_batch_size, 
                 test_batch_size, 
                 num_workers, 
                 pin_memory: bool = True) -> None:
        super().__init__()
        #self.task = task
        self.image_folder_path = image_folder_path
        self.rg_folder_path = rg_folder_path
        self.image_csv_path = image_csv_path
        self.rg_csv_path = rg_csv_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        

    def setup(self, stage: [str] = None) -> None:
        """Setup the data module.

        Args:
            stage (Optional[str], optional): Stage of the data module setup. Defaults to None.
        """
        if stage == "fit" or stage is None:
            airogs_full = AirogsDataset(task="classification", 
                                        image_folder_path=self.image_folder_path, 
                                        rg_folder_path=self.rg_folder_path, 
                                        image_csv_path=self.image_csv_path, 
                                        rg_csv_path=self.rg_csv_path)
            train_size = int(0.8 * len(airogs_full))
            val_size = len(airogs_full) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(airogs_full, [train_size, val_size])
        if stage == "test" or stage is None:
            self.test_dataset = AirogsDataset(task="classification", 
                                              image_folder_path=self.image_folder_path, 
                                              rg_folder_path=self.rg_folder_path, 
                                              image_csv_path=self.image_csv_path, 
                                              rg_csv_path=self.rg_csv_path)

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