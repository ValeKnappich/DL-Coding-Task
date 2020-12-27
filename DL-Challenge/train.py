from data import DataModule
from model import IntentAndEntityModel
from config import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import GPUtil


def main():
    data_module = DataModule()
    model = IntentAndEntityModel()

    gpus = GPUtil.getAvailable(order="first", limit=1, maxLoad=0.3, maxMemory=0.3)
    print(f"Training on GPUs: {gpus}")

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=20,
        precision=16,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        checkpoint_callback=False,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(config.ckpt_path)


if __name__ == "__main__":
    main()