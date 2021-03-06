<a name="data"></a>
# data

<a name="data.NLUDataSet"></a>
## NLUDataSet Objects

```python
class NLUDataSet(Dataset)
```

Generic Dataset, that holds a dict with columns

<a name="data.NLUDataSet.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(data: dict)
```

NLUDataset constructor. Converts the columns to tensors of type long.

**Arguments**:

- `data` _dict_ - Dict holding the columns.

<a name="data.DataModule"></a>
## DataModule Objects

```python
class DataModule(pl.LightningDataModule)
```

Data Module to load data from disk and create DataLoaders

<a name="data.DataModule.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(mode: str = "fit", no_split: bool = False)
```

DataModule constructor. Loads config from config.py and binds them to the object.
Calls the setup method to load data for the given mode (see below).

**Arguments**:

- `mode` _str, optional_ - Specify to load either training or testing data ("fit" or "test"). Defaults to "fit".
- `no_split` _bool, optional_ - Flag to not perform a random split. Defaults to False.

<a name="data.DataModule.load_and_clean"></a>
#### load\_and\_clean

```python
 | load_and_clean() -> dict
```

Load data from disk and remove characters that are not supported by the tokenizer

**Returns**:

- `data` _dict_ - dict containing the columns as long tensors

<a name="data.DataModule.setup"></a>
#### setup

```python
 | setup(mode: str)
```

Load data from disk and create NLUDataSets.
Uses the pretrained tokenizer (according to config.config.model_name),
uses padding and truncation to get a constant sequence length (config.config.sequence_length).

**Arguments**:

- `mode` _str_ - Specify to load either training or testing data ("fit" or "test").
  

**Raises**:

- `ValueError` - If mode is not "fit" or "test".

<a name="data.DataModule.train_dataloader"></a>
#### train\_dataloader

```python
 | train_dataloader() -> DataLoader
```

Create DataLoader from train set.

**Returns**:

- `DataLoader` - Training Loader

<a name="data.DataModule.val_dataloader"></a>
#### val\_dataloader

```python
 | val_dataloader() -> DataLoader
```

Create DataLoader from validation set.

**Returns**:

- `DataLoader` - Validation Loader

<a name="data.DataModule.test_dataloader"></a>
#### test\_dataloader

```python
 | test_dataloader() -> DataLoader
```

Create DataLoader from test set.

**Returns**:

- `DataLoader` - Test Loader

<a name="data.DataModule.format2IOB"></a>
#### format2IOB

```python
 | format2IOB(tokens_matrix: torch.Tensor) -> torch.Tensor
```

Convert the format from json containig entity spans to IOB format.

**Arguments**:

- `tokens_matrix` _torch.Tensor_ - Tokens created by the tokenizer of shape (num_examples, sequence_length)
  

**Returns**:

- `torch.Tensor` - Token labels of shape (num_examples, sequence_length)

<a name="data.DataModule.format2original"></a>
#### format2original

```python
 | format2original(intents: torch.Tensor, token_labels_matrix: torch.Tensor) -> dict
```

Format back to the original JSON format.

**Arguments**:

- `intents` _torch.Tensor_ - Tensor of the predicted intent-ID's
- `token_labels_matrix` _torch.Tensor_ - Tensor of the tokens
  

**Returns**:

- `dict` - results in the original JSON format

