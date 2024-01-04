import sys
sys.path.append('/work/scratch/tyang/new_ensemble_model')

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as T
from PIL import Image

from pytorch_lightning.loggers import logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from new_ensemble_model.ensemble_model.models.EffientNetB4 import EfficientNetB4Module
from new_ensemble_model.ensemble_model.data.datamodule import AirogsDataModule


model = EfficientNetB4Module()

datamodule = AirogsDataModule(image_folder_path="/work/scratch/tyang/yolov5results",
                              
                              image_csv_path="/images/innoretvision/eye/airogs/train_labels.csv",
                              
                              transform=T.Compose([
                                    T.ToPILImage(),
                                    
                                    T.Resize((384,384), interpolation=Image.BICUBIC),
                                    T.CenterCrop(380),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                ]),
                              train_batch_size=32, 
                              test_batch_size=32, 
                              num_workers=8)

datamodule.setup()


early_stopping_callback = EarlyStopping(
    monitor="val/f1",
    mode="max",
    patience=150,
    verbose=True,
    
)
checkpoint_callback = ModelCheckpoint(
    dirpath="/work/scratch/tyang/new_ensemble_model/ensemble_model/checkpoints",
    filename="best-checkpoint",
    monitor="val/f1",
    mode="max",
    
)

trainer = Trainer(
    max_epochs=300,
    gpus=1,
    callbacks=[early_stopping_callback, checkpoint_callback],
)

trainer.fit(model, datamodule)