import sys
sys.path.append('/work/scratch/tyang')
from .data.airogs import AirogsDataModule
from .data.airogs import AirogsDataset
from .models.deit import DeiTModule
from .models.EffientNetB4 import EfficientNetB4Module
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def load_callbacks():
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val/f1', 
                                   patience=10, 
                                   mode='max',
                                   min_delta=0.001))
    
    callbacks.append(ModelCheckpoint(monitor='val/f1', 
                                     dir_path="/work/scratch/tyang/checkpoints",
                                     filename=" effientnetb4-{epoch:02d}-{val/f1:.2f}",
                                     save_top_k=1, 
                                     mode='max', 
                                     ))
    return callbacks




def main(args):
    dict_args = vars(args)

 
    if args.model_name == "DeiTModule":
        model = DeiTModule(**dict_args)

    if args.model_name == "EffientNetB4Module":
        model = EfficientNetB4Module(**dict_args)

    if args.dataset_name == "Airogs":
        datamodule = AirogsDataModule(**dict_args)
    

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

  
    parser.add_argument("--model_name", type=str, default="EffcientNetB4Module", help="model name")
    parser.add_argument("--dataset_name", type=str, default="Airogs", help="dataset name")
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_name == "DeiTModule":
        parser = DeiTModule.add_model_specific_args(parser) 
    
    if temp_args.model_name == "EffientNetB4Module":
        parser = EfficientNetB4Module.add_model_specific_args(parser)
    
    if temp_args.dataset_name == "Airogs":
        parser = AirogsDataModule.add_dataset_specific_args(parser)

    args = parser.parse_args()

    
    main(args)

    