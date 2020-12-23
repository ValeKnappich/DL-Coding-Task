from config import config

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer, BertModel

class IntentAndEntityBERT(pl.LightningModule):

  def __init__(self, sequence_length, num_intents, num_entities):
    super().__init__()
    self.batch_size = config["batch_size"]
    self.bert_lr = config["bert_lr"]
    self.head_lr = config["head_lr"]
    self.model_name = config["model_name"]
    self.sequence_length = sequence_length

    self.bert = BertModel.from_pretrained(self.model_name)
    self.linear_intent = nn.Linear(self.sequence_length * 768, num_intents)
    self.linear_ner = nn.Linear(768, num_entities * 2 + 1)


  def forward(self, input_ids, attention_mask):
    batch_size = input_ids.shape[0] # = self.batch_size except for last batch
    bert_output = self.bert(input_ids, attention_mask)

    intent_input = bert_output.last_hidden_state.view(batch_size, -1)
    intent_logits = self.linear_intent(intent_input)
    
    # process all tokens at the same time as if the batch was batch_size * sequence length
    ner_input = bert_output.last_hidden_state.view(batch_size * self.sequence_length, -1)
    ner_logits = self.linear_ner(ner_input).view(batch_size, self.sequence_length, -1)
    return intent_logits, ner_logits


  def configure_optimizers(self):
    opt = torch.optim.Adam(
        [{"params": self.bert.parameters(), "lr": self.bert_lr},
         {"params": self.linear_intent.parameters(), "lr": self.head_lr},
         {"params": self.linear_ner.parameters(), "lr": self.head_lr}]
    )
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=[
            lambda epoch: 1,
            lambda epoch: 1 if epoch <= 2 else 0.9**epoch,
            lambda epoch: 1
        ]                                          
    )
    return [opt], [lr_sched]


  def loss(self, intent_logits, ner_logits, intent_labels, ner_labels):
    intent_loss = F.cross_entropy(intent_logits,  intent_labels)
    ner_loss = F.cross_entropy( # process all tokens at once as bigger batch
        ner_logits.view(ner_logits.shape[0] * self.sequence_length, -1), 
        ner_labels.view(-1)
    )
    return intent_loss, ner_loss


  def accuracy(self, logits, labels):
    labels_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    correct = labels_pred == labels
    return sum(correct) / len(correct)


  def training_step(self, batch, batch_idx):
    intent_logits, ner_logits = self.forward(batch["input_ids"], batch["attention_mask"])
    intent_loss, ner_loss = self.loss(intent_logits, ner_logits, batch["intent"], batch["token_labels"])
    combined_loss = (intent_loss + ner_loss) / 2
    intent_acc = self.accuracy(intent_logits, batch["intent"])
    ner_acc = self.accuracy(
        ner_logits.view(ner_logits.shape[0] * self.sequence_length, -1), 
        batch["token_labels"].view(-1)
    )
    self.log("train_loss", combined_loss)
    self.log("train_intent_loss", intent_loss)
    self.log("train_ner_loss", ner_loss)
    self.log("train_intent_acc", intent_acc)
    self.log("train_ner_acc", ner_acc)
    return combined_loss


  def validation_step(self, batch, batch_idx):
    intent_logits, ner_logits = self.forward(batch["input_ids"], batch["attention_mask"])
    intent_loss, ner_loss = self.loss(intent_logits, ner_logits, batch["intent"], batch["token_labels"])
    combined_loss = (intent_loss + ner_loss) / 2
    intent_acc = self.accuracy(intent_logits, batch["intent"])
    ner_acc = self.accuracy(
        ner_logits.view(ner_logits.shape[0] * self.sequence_length, -1), 
        batch["token_labels"].view(-1)
    )
    self.log("val_loss", combined_loss)
    self.log("val_intent_loss", intent_loss)
    self.log("val_ner_loss", ner_loss)
    self.log("val_intent_acc", intent_acc)
    self.log("val_ner_acc", ner_acc)