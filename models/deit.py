from typing import Any, Dict, List, Optional
import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryF1Score
import timm
import torch.nn.functional as F
from ..utils.Focal_Loss import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
import torch.optim.lr_scheduler as lrs
import argparse


class DeiTModule(LightningModule):
    def __init__(self,
        
                freeze,
                dropout,
                learning_rate,
                weight_decay,
                **kwargs,
   ):
                super().__init__()

                # this line allows to access init params with 'self.hparams' attribute
                # also ensures init params will be stored in ckpt
                self.save_hyperparameters()

        
                self.model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True)
                
                if freeze:
                    for param in self.model.parameters():
                            param.requires_grad = False
                self.model.head = torch.nn.Sequential(
                    torch.nn.Dropout(dropout, inplace=False),
                    torch.nn.Linear(self.model.head.in_features, 1)
                )
                
                self.model.head_dist = torch.nn.Sequential(
                    torch.nn.Dropout(dropout, inplace=False),
                    torch.nn.Linear(self.model.head_dist.in_features, 1),
                )
                
                self.criterion = FocalLoss()
                self.f1_score = BinaryF1Score()
                self.AdamW = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeiTModule")
        parser.add_argument("--freeze", type=bool, default=False)
        parser.add_argument("--dropout", type=int, default=0.3)
        parser.add_argument("--learning_rate", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        return parent_parser



                 
    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        outlayer = torch.nn.Sigmoid()
        x = outlayer(x)
        return x
        
    
    def model_step(self, batch: Any):
        x, y = batch
        predit = self.forward(x)

        y = y.to(torch.float)

        if predit.shape != y.shape:
            predit = predit.squeeze(1)

        loss = self.criterion(predit, y)
        return loss, predit, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.f1_score(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.f1_score(preds, targets), on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `trainicriterionng_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        _, y = batch
        loss, preds, targets = self.model_step(batch)

        
        f1_score = self.f1_score(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        self.log("val/f1", f1_score, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": y,
                "f1": f1_score}

    def validation_epoch_end(self, outputs: List[Any]):
        f1_score = torch.stack([x["f1"] for x in outputs]).mean()
        self.log("val/f1", f1_score)
        return {"val/f1": f1_score.item()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
         
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
       
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

   
    def configure_optimizers(self):
        optimizer = self.AdamW
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6,)
        return [optimizer], [scheduler]