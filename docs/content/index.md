# DL Coding Task

This repository contains my solution to the deep learning coding task from Prof. Vu.

## Scripts

- **config.py**: File that contains constants like UNIQUE_INTENTS and mappings to IDs. Also contains hyperparameters.
- **model.py**: Defines the model as PyTorch Lightning Module and therefore implements the model architecture, as well as training and validation procedure. Model is saved after training.
- **train.py**: Used to train the model with a pl.Trainer. Uses multiple GPU's if available and performs early stopping.
- **data.py**: Defines the pl.LightningDataModule, that implements data loading and preprocessing. Executing the script will run and end-to-end test, weather the conversion between formats worked correctly for the training data.
- **test.py**: Used to perform inference on the dev set.

## Usage

```bash
python train.py         # train the model, checkpoint is saved to disk
python test.py          # loads the dumped model, predicts the classes and outputs the results to disk
```

