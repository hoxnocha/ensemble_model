import sys
sys.path.append('/work/scratch/tyang/new_ensemble_model')

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as T  
from PIL import Image
from pytorch_lightning.loggers import logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from new_ensemble_model.ensemble_model.models.EfficientNetV2 import EfficientNetV2Module
from new_ensemble_model.ensemble_model.data.datamodule import AirogsDataModule
#from data.dataset import AirogsDataset

model = EfficientNetV2Module()

datamodule = AirogsDataModule(image_folder_path="/work/scratch/tyang/yolov5results",
                              
                              image_csv_path="/images/innretvision/eye/airogs/train_labels.csv",
                             
                              transform=T.Compose([
                                    T.ToPILImage(),
                                    
                                    T.Resize((480,480), interpolation=Image.BICUBIC),
                                    T.CenterCrop(480),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.5, 0.5, 0.5], 
                                                std=[0.5, 0.5, 0.5])
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
    filename="enetv2-checkpoint",
    monitor="val/f1",
    mode="max",
    
)

trainer = Trainer(
    max_epochs=300,
    gpus=1,
    callbacks=[early_stopping_callback, checkpoint_callback],
)

trainer.fit(model, datamodule)