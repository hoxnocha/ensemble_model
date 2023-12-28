from typing import Any, Dict, List, Optional
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import BinaryF1Score
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import torch.nn.functional as F

#from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

class EfficientNetModule(LightningModule):
    def __init__(self,
                 #optimizer: torch.optim.Optimizer, 
                 #scheduler: torch.optim.lr_scheduler 
        ):
                 super().__init__()
                 self.save_hyperparameters()
        
                 self.model = EfficientNet.from_pretrained('efficientnet-b4')
         
                 in_features = self.model._fc.in_features
                 #self.model._fc = torch.nn.Linear(in_features, 1)
                 # = torch.tensor([1.0, 30.0])
                 self.criterion = torch.nn.BCEWithLogitsLoss()
        #layers = list(backbone.children())[:-2]
        #self.feature_extractor = torch.nn.Sequential(*layers)
                 self.model._fc= torch.nn.Sequential(
                  torch.nn.Linear(in_features, 128),
                  torch.nn.ReLU(),
                  torch.nn.Dropout(p=0.4, inplace=False),
                  torch.nn.Linear(128, 1),
                )
        #self.classifier = torch.nn.Linear(in_features, 2)
                 #self.train_acc = Accuracy(task='binary', num_classes=1)
                 #self.val_acc = Accuracy(task='binary', num_classes=1)
                 #self.test_acc = Accuracy(task='binary', num_classes=1)
                 self.f1_score = BinaryF1Score()

                 self.train_loss = MeanMetric()
                 self.val_loss = MeanMetric()
                 self.test_loss = MeanMetric()
                 
                 
                 
    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        outlayer = torch.nn.Sigmoid()
        x = outlayer(x)
        return x
        
    def on_train_start(self) -> None:
         # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        #self.val_acc.reset()
        #self.val_acc_best.reset()

    

    
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
        print(loss, preds, targets)

        # update and log metrics
        self.train_loss(loss)
        self.f1_score(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.f1_score(preds, targets), on_step=False, on_epoch=True, prog_bar=True)

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

        # update and log metrics
        self.val_loss(loss)
        f1_score = self.f1_score(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("val/f1", f1_score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": y,
                "f1": f1_score}

    def validation_epoch_end(self, outputs: List[Any]):
        f1_score = torch.stack([x["f1"] for x in outputs]).mean()
        self.log("val/f1", f1_score)
        return {"val/f1": f1_score.item()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        
        print(preds, targets)

        # update and log metrics
        self.test_loss(loss)
        #self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


#if __name__ == "__main__":
 #   import pdb; pdb.set_trace()
  #  model = EfficientNetModule()
   # print("end")

