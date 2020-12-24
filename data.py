from config import *

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import tokenizations as tk
from sklearn.model_selection import train_test_split

import pandas as pd
import json
from math import ceil
from warnings import warn


class NLUDataSet(Dataset):

  def __init__(self, data):
    self.data = {field: torch.Tensor(values).long() for field, values in data.items()}

  def __len__(self):
    return len(self.data["input_ids"])

  def __getitem__(self, idx):
    return {field: self.data[field][idx] for field in self.data}



class BERTDataModule(pl.LightningDataModule):

  def __init__(self, mode="fit", no_split=False):
    super().__init__()
    self.batch_size = config.batch_size
    self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
    self.path = config.train_path if mode == "fit" else config.dev_path
    self.test_size = config.test_size if not no_split else None
    self.sequence_length = config.sequence_length
    self.setup(mode)

  def setup(self, mode):
    # Load data from disk
    data_dict = json.load(open(self.path, "r"))
    self.data_list = [data_dict[i] for i in data_dict]
    # input_ids, attention_mask, intents, token_labels
    encodings = self.tokenizer(
        [instance["text"] for instance in self.data_list], 
        padding="max_length", truncation=True, max_length=self.sequence_length
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    # training mode
    if mode == "fit":
      intents = [INTENT2ID[instance["intent"]] for instance in self.data_list]
      tokens = [self.tokenizer.tokenize(instance["text"]) for instance in self.data_list] 
      token_labels = self.format2IOB(tokens)
      data = {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
          "intent": intents,
          "token_labels": token_labels
      }
      if self.test_size:
        # split data and create loaders
        data_df = pd.DataFrame(data) # use df to make split on column based format
        train_df, val_df = train_test_split(data_df, test_size=self.test_size)
        self.train = NLUDataSet(train_df.to_dict(orient="list"))
        self.val = NLUDataSet(val_df.to_dict(orient="list"))
      else:
        self.train = NLUDataSet(data)
    # test mode
    elif mode == "test":
      data = {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
      }
      self.test = NLUDataSet(data)
    else:
      raise ValueError("Invalid mode " + mode)


  def train_dataloader(self):
    return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)


  def val_dataloader(self):
    return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4)


  def test_dataloader(self):
    return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4)


  def format2IOB(self, bert_tokens_matrix):
    token_labels = []
    for instance, bert_tokens in zip(self.data_list, bert_tokens_matrix):
      # Initialize with outside tokens
      token_labels_sentence = [ENTITY2ID["O"] for _ in range(self.sequence_length)]
      # Iterate over entities present in the sentence
      for entity, (start, end) in instance["positions"].items():
        # Calculate offsets  by counting spaces and #'s
        num_whitespaces_before = instance["text"][:start].count(" ")
        start_prime = start - num_whitespaces_before
        char_count = 0
        # Iterate over bert tokens and check if index is reached
        for i, bert_token in enumerate(bert_tokens):
          char_count += len(bert_token) - bert_token.count("#")
          if char_count > start_prime:
            # Assign begin token
            token_labels_sentence[i] = ENTITY2ID[f"B-{entity}"]
            # Check how many inside tokens there are
            len_prime = end - start - instance["text"][start:end].count(" ") # entity length without whitespaces
            char_count = len(bert_token) - bert_token.count("#")
            token_count = 0
            while char_count <= len_prime and i + token_count + 1 < len(bert_tokens):
              char_count += len(bert_tokens[i + token_count + 1]) - bert_tokens[i + token_count + 1].count("#")
              token_count += 1
            inside_token = ENTITY2ID[f"I-{entity}"]
            token_labels_sentence[i+1:i+token_count+1] = [inside_token for _ in range(token_count)]
            break
      token_labels.append(token_labels_sentence)
    return token_labels


  def _get_entities(self, token_labels):
    is_inside = lambda l: l % 2 == 1
    is_outside = lambda l: l == ENTITY2ID["O"]
    is_begin = lambda l: l % 2 == 0 and not l == ENTITY2ID["O"]
    entities = []
    current = [] # [label, start, stop]
    for i, token_label in enumerate(token_labels):
      if not current and is_begin(token_label):
        current = [token_label, i, i]
      elif current and is_begin(token_label):
        if token_label == current[1]:
          print("Begin Token inside the Entity")
          current[2] = i
        else:
          entities.append(current)
          current = [token_label, i, i]
      elif not current and is_inside(token_label):
        print("Entity begins with Inside Token")
        current = [token_label - 1, i, i]
      elif current and is_inside(token_label):
        current[2] = i
      elif current and is_outside(token_label):
        entities.append(current)
        current = []
    # check for double entries
    if len(set([e[0] for e in entities])) != len(entities):
      warn("Multiple entries for the same entity")  
    return entities
    
   
  def format2original(self, intents, token_labels_matrix):
    result = {}
    intents = [ID2INTENT[intent] for intent in intents]
    for i, (intent, token_labels, raw) in enumerate(zip(intents, token_labels_matrix, self.data_list)):
      ws_tokens = raw["text"].split()
      bert_tokens = self.tokenizer.tokenize(raw["text"])
      ws2bert, bert2ws = tk.get_alignments(ws_tokens, bert_tokens)
      entities = self._get_entities(token_labels)
      for j, (label, start, end) in enumerate(entities):
        chars_before = sum([len(t) - t.count("#") for t in bert_tokens[:start]])
        try:
          ws_before = len(ws_tokens[:min(bert2ws[start])])
          char_start = chars_before + ws_before
          ws_inside = len(ws_tokens[min(bert2ws[start]):max(bert2ws[end]) + 1]) - 1
          char_end = char_start + sum([len(t) - t.count("#") for t in bert_tokens[start:end+1]]) + ws_inside
          entities[j][1] = char_start
          entities[j][2] = char_end
        except ValueError as e:
          warn("Could not extract entity, empty alignment")
          continue
        except IndexError as e:
          warn("Could not extract entities, overlapping ranges")
          continue

      result[str(i)] = {
        "text": raw["text"], "intent": intent,
        "positions": {
          ID2ENTITY[label][2:]: [start, end-1]
          for label, start, end in entities
        },
        "slots": {
          ID2ENTITY[label][2:]: raw["text"][start:end]
          for label, start, end in entities
        }
      }
    return result
  

if __name__ == "__main__":
  # End to end test for data conversion
  dm = BERTDataModule(mode="fit", no_split=True)
  orig = {
    str(i): v for i, v in enumerate(dm.data_list)
  }
  reconstruct = dm.format2original(dm.train.data["intent"].tolist(), dm.train.data["token_labels"].tolist())
  errors = [(orig[str(i)], reconstruct[str(i)]) for i in range(len(reconstruct)) if orig[str(i)] != reconstruct[str(i)]]
  inspect = input("Number of Errors: " + str(len(errors)) + "y/n")
  if inspect == "y":
    for i in range(len(reconstruct)):
      if orig[str(i)] != reconstruct[str(i)]:
        print(orig[str(i)])
        print(reconstruct[str(i)])
        print("\n")
        import pdb; pdb.set_trace()