from data import BERTDataModule
from model import IntentAndEntityBERT
from config import config

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

data_module = BERTDataModule()
# model = IntentAndEntityBERT(
#     data_module.sequence_length, 
#     len(data_module.unique_intents),
#     len(data_module.unique_entities),
# )
# trainer = pl.Trainer(
#     gpus=[2], max_epochs=20, precision=16, 
#     callbacks=[EarlyStopping(monitor='val_loss')], 
#     checkpoint_callback=False,
#     progress_bar_refresh_rate=2
# )
# trainer.fit(model, datamodule=data_module)