from data import DataModule
from model import IntentAndEntityModel
from config import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import GPUtil

data_module = DataModule()

model = IntentAndEntityModel(
    config.sequence_length, 
    len(UNIQUE_INTENTS),
    len(UNIQUE_ENTITIES),
)

trainer = pl.Trainer(
    gpus=GPUtil.getAvailable(order='first', limit=4, maxLoad=0.3, maxMemory=0.3), 
    max_epochs=20, precision=16, 
    callbacks=[EarlyStopping(monitor='val_loss', mode='min')], 
    checkpoint_callback=False
)
trainer.fit(model, datamodule=data_module)
trainer.save_checkpoint(config.ckpt_path)