<a name="data"></a>
# data

<a name="data.DataModule"></a>
## DataModule Objects

```python
class DataModule(pl.LightningDataModule)
```

<a name="data.DataModule.load_and_clean"></a>
#### load\_and\_clean

```python
 | load_and_clean() -> dict
```

Load data from disk and remove characters that are not supported by the tokenizer

**Returns**:

- `dict` - [description]

