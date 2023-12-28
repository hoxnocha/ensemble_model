import sys
sys.path.append('/work/scratch/tyang')

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


from pytorch_lightning.loggers import logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.lightning_model import EfficientNetModule
from data.datamodule import AirogsDataModule
#from data.dataset import AirogsDataset

model = EfficientNetModule()

datamodule = AirogsDataModule(image_folder_path="/home/students/tyang/yolov5/runs/detect/cat0/crops/od",
                              rg_folder_path="/home/students/tyang/yolov5/runs/detect/RG_images/crops/od",
                              image_csv_path="/home/students/tyang/airogs/train_labels.csv",
                              rg_csv_path="/home/students/tyang/yolov5/runs/detect/RG_images/crops/rg.csv",
                              train_batch_size=32, 
                              test_batch_size=32, 
                              num_workers=8)

datamodule.setup()


early_stopping_callback = EarlyStopping(
    monitor="val/f1",
    mode="max",
    patience=100,
    verbose=True,
    
)
checkpoint_callback = ModelCheckpoint(
    dirpath="home/students/tyang/ensemble_model/checkpoints",
    filename="best-checkpoint",
    monitor="val/f1",
    mode="max",
    
)

trainer = Trainer(
    max_epochs=300,
    gpus=1,
    callbacks=[early_stopping_callback,checkpoint_callback],
)

trainer.fit(model, datamodule)