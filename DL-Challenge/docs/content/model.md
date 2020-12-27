<a name="model"></a>
# model

<a name="model.IntentAndEntityModel"></a>
## IntentAndEntityModel Objects

```python
class IntentAndEntityModel(pl.LightningModule)
```

<a name="model.IntentAndEntityModel.forward"></a>
#### forward

```python
 | forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Forward pass

**Arguments**:

- `input_ids` _torch.Tensor_ - [description]
- `attention_mask` _torch.Tensor_ - [description]
  

**Returns**:

  Tuple[torch.Tensor, torch.Tensor]: [description]

<a name="model.IntentAndEntityModel.configure_optimizers"></a>
#### configure\_optimizers

```python
 | configure_optimizers() -> Tuple[
 |         List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
 |     ]
```

Configure optimizer and lr scheduler

**Returns**:

  Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: [description]

<a name="model.IntentAndEntityModel.loss"></a>
#### loss

```python
 | loss(intent_logits: torch.Tensor, ner_logits: torch.Tensor, intent_labels: torch.Tensor, ner_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Calculate the individual losses for intent and ner respectively

**Arguments**:

- `intent_logits` _torch.Tensor_ - [description]
- `ner_logits` _torch.Tensor_ - [description]
- `intent_labels` _torch.Tensor_ - [description]
- `ner_labels` _torch.Tensor_ - [description]
  

**Returns**:

  Tuple[torch.Tensor, torch.Tensor]: [description]

<a name="model.IntentAndEntityModel.accuracy"></a>
#### accuracy

```python
 | accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float
```

Calculate accuracy for logits and labels

**Arguments**:

- `logits` _torch.Tensor_ - [description]
- `labels` _torch.Tensor_ - [description]
  

**Returns**:

- `float` - [description]

<a name="model.IntentAndEntityModel.training_step"></a>
#### training\_step

```python
 | training_step(batch: dict, batch_idx: int) -> torch.Tensor
```

Training step

**Arguments**:

- `batch` _dict_ - [description]
- `batch_idx` _int_ - [description]
  

**Returns**:

- `torch.Tensor` - [description]

<a name="model.IntentAndEntityModel.validation_step"></a>
#### validation\_step

```python
 | validation_step(batch: dict, batch_idx: int)
```

Validation step

**Arguments**:

- `batch` _dict_ - [description]
- `batch_idx` _int_ - [description]

