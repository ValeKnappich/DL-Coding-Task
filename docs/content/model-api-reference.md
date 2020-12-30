<a name="model"></a>
# model

<a name="model.IntentAndEntityModel"></a>
## IntentAndEntityModel Objects

```python
class IntentAndEntityModel(pl.LightningModule)
```

<a name="model.IntentAndEntityModel.__init__"></a>
#### \_\_init\_\_

```python
 | __init__()
```

IntentAndEntityModel constructor. Loads the config from config.py and binds it to the object.
Create Layers, including pretrained, as specified in config.config.model_name

<a name="model.IntentAndEntityModel.forward"></a>
#### forward

```python
 | forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Forward pass

**Arguments**:

- `input_ids` _torch.Tensor_ - Input ID's created by the tokenizer of shape (batch_size, sequence_length)
- `attention_mask` _torch.Tensor_ - Attention masks created by the tokenizer of shape (batch_size, sequence_length)
  

**Returns**:

  intent_logits, ner_logits (Tuple[torch.Tensor, torch.Tensor]): Logits of the intent and NER heads before Softmax

<a name="model.IntentAndEntityModel.configure_optimizers"></a>
#### configure\_optimizers

```python
 | configure_optimizers() -> Tuple[
 |         List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
 |     ]
```

Configure optimizer and lr scheduler

**Returns**:

  Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: Optimizer and LR Scheduler

<a name="model.IntentAndEntityModel.loss"></a>
#### loss

```python
 | loss(intent_logits: torch.Tensor, ner_logits: torch.Tensor, intent_labels: torch.Tensor, ner_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Calculate the individual losses for intent and ner respectively

**Arguments**:

- `intent_logits` _torch.Tensor_ - Intent logits before SoftMax
- `ner_logits` _torch.Tensor_ - NER logits before SoftMax
- `intent_labels` _torch.Tensor_ - Predicted intent-ID's
- `ner_labels` _torch.Tensor_ - Predicted token labels
  

**Returns**:

  Tuple[torch.Tensor, torch.Tensor]: Intent loss and NER loss

<a name="model.IntentAndEntityModel.accuracy"></a>
#### accuracy

```python
 | accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float
```

Calculate accuracy for logits and labels

**Arguments**:

- `logits` _torch.Tensor_ - Logits
- `labels` _torch.Tensor_ - True labels
  

**Returns**:

- `float` - Accuracy

<a name="model.IntentAndEntityModel.training_step"></a>
#### training\_step

```python
 | training_step(batch: dict, batch_idx: int) -> torch.Tensor
```

Training step

**Arguments**:

- `batch` _dict_ - Dict containing the columns of the batch
- `batch_idx` _int_ - Batch index
  

**Returns**:

- `torch.Tensor` - Combined loss as mean between intent and NER loss

<a name="model.IntentAndEntityModel.validation_step"></a>
#### validation\_step

```python
 | validation_step(batch: dict, batch_idx: int)
```

Validation step

**Arguments**:

- `batch` _dict_ - Dict containing the columns of the batch
- `batch_idx` _int_ - Batch index

