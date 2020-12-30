from config import *

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import tokenizations as tk
from sklearn.model_selection import train_test_split

import pandas as pd
import json
from math import ceil
from warnings import warn
import re


class NLUDataSet(Dataset):
    """Generic Dataset, that holds a dict with columns"""

    def __init__(self, data: dict):
        """NLUDataset constructor. Converts the columns to tensors of type long.

        Args:
            data (dict): Dict holding the columns.
        """
        self.data = {
            field: torch.Tensor(values).long() for field, values in data.items()
        }

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {field: self.data[field][idx] for field in self.data}


class DataModule(pl.LightningDataModule):
    """Data Module to load data from disk and create DataLoaders"""

    def __init__(self, mode: str = "fit", no_split: bool = False):
        """DataModule constructor. Loads config from config.py and binds them to the object.
        Calls the setup method to load data for the given mode (see below).

        Args:
            mode (str, optional): Specify to load either training or testing data ("fit" or "test"). Defaults to "fit".
            no_split (bool, optional): Flag to not perform a random split. Defaults to False.
        """
        super().__init__()
        self.batch_size = config.batch_size
        self.pretrained_model_name = config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.path = config.train_path if mode == "fit" else config.dev_path
        self.test_size = config.test_size if not no_split else None
        self.sequence_length = config.sequence_length
        self.replace_char = config.replace_char
        self.setup(mode)

    def load_and_clean(self) -> dict:
        """Load data from disk and remove characters that are not supported by the tokenizer

        Returns:
            data (dict): dict containing the columns as long tensors
        """

        def replace_all(s, unsupported, replacement=self.replace_char):
            for unsup_char in unsupported:
                s = s.replace(unsup_char, replacement)
            return s

        data_dict = json.load(open(self.path, "r"))
        # Get unique characters
        chars = set("".join(i["text"] for i in data_dict.values()))
        # Get unsupported chars by comparing it with tokenizer output
        unsupported_chars = []
        for char in chars:
            tok_char = self.tokenizer.tokenize(char)
            if len(tok_char) != 1 or tok_char[0] != char:
                unsupported_chars.append(char)
        if " " in unsupported_chars:
            unsupported_chars.remove(" ")
        self.unsupported_chars = set(unsupported_chars)
        # Clean dict
        for i in data_dict:
            data_dict[i]["uncleaned"] = data_dict[i]["text"]
            data_dict[i]["text"] = replace_all(
                data_dict[i]["text"], self.unsupported_chars
            )
        return data_dict

    added_chars = {
        # chars that are added by the respective tokenizer
        "bert-base-uncased": "#",
        "roberta-base": "Ġ",
    }

    def setup(self, mode: str):
        """Load data from disk and create NLUDataSets.
        Uses the pretrained tokenizer (according to config.config.model_name),
        uses padding and truncation to get a constant sequence length (config.config.sequence_length).

        Args:
            mode (str): Specify to load either training or testing data ("fit" or "test").

        Raises:
            ValueError: If mode is not "fit" or "test".
        """
        # Load data from disk
        data_dict = self.load_and_clean()
        self.data_list = [data_dict[i] for i in data_dict]
        encodings = self.tokenizer(
            [instance["text"] for instance in self.data_list],
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        # training mode
        if mode == "fit":
            intents = [INTENT2ID[instance["intent"]] for instance in self.data_list]
            tokens = [
                self.tokenizer.tokenize(instance["text"]) for instance in self.data_list
            ]
            tokens = [
                tok[: self.sequence_length] for tok in tokens
            ]  # truncate to max length
            token_labels = self.format2IOB(tokens)
            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "intent": intents,
                "token_labels": token_labels,
            }
            if self.test_size:
                # split data and create loaders
                data_df = pd.DataFrame(
                    data
                )  # use df to make split on column based format
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

    def train_dataloader(self) -> DataLoader:
        """Create DataLoader from train set.

        Returns:
             DataLoader: Training Loader
        """
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader from validation set.

        Returns:
             DataLoader: Validation Loader
        """
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader from test set.

        Returns:
             DataLoader: Test Loader
        """
        return DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def format2IOB(self, tokens_matrix: torch.Tensor) -> torch.Tensor:
        """Convert the format from json containig entity spans to IOB format.

        Args:
            tokens_matrix (torch.Tensor): Tokens created by the tokenizer of shape (num_examples, sequence_length)

        Returns:
            torch.Tensor: Token labels of shape (num_examples, sequence_length)
        """
        added_char = self.added_chars[self.pretrained_model_name]
        token_labels = []
        for instance, tokens in zip(self.data_list, tokens_matrix):
            # Initialize with outside tokens
            token_labels_sentence = [
                ENTITY2ID["O"] for _ in range(self.sequence_length)
            ]
            # Iterate over entities present in the sentence
            for entity, (start, end) in instance["positions"].items():
                # Calculate offsets  by counting spaces and #'s
                num_whitespaces_before = instance["text"][:start].count(" ")
                start_prime = start - num_whitespaces_before
                char_count = 0
                # Iterate over tokens and check if index is reached
                for i, token in enumerate(tokens):
                    char_count += len(token) - token.count(added_char)
                    if char_count > start_prime:
                        # Assign begin token
                        token_labels_sentence[i] = ENTITY2ID[f"B-{entity}"]
                        # Check how many inside tokens there are
                        len_prime = (
                            end - start - instance["text"][start:end].count(" ")
                        )  # entity length without whitespaces
                        char_count = len(token) - token.count(added_char)
                        token_count = 0
                        while char_count <= len_prime and i + token_count + 1 < len(
                            tokens
                        ):
                            char_count += len(tokens[i + token_count + 1]) - tokens[
                                i + token_count + 1
                            ].count(added_char)
                            token_count += 1
                        inside_token = ENTITY2ID[f"I-{entity}"]
                        token_labels_sentence[i + 1 : i + token_count + 1] = [
                            inside_token for _ in range(token_count)
                        ]
            token_labels.append(token_labels_sentence)
        return token_labels

    def _get_entities(self, token_labels: torch.Tensor) -> List[list]:
        """Get entities from token labels.

        Args:
            token_labels (torch.Tensor): token labels created by format2IOB.

        Returns:
            List[list]: List of lists of format [label, start, stop] where label is the encoded beginning token.
        """
        is_inside = lambda l: l % 2 == 1
        is_outside = lambda l: l == ENTITY2ID["O"]
        is_begin = lambda l: l % 2 == 0 and not l == ENTITY2ID["O"]
        entities = []
        current = []  # [label, start, stop]
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

    def _find_best_match(
        self, matches: List[Tuple[int, int]], tokens: list, t_start: int, t_end: int
    ) -> Tuple[int, int]:
        """Find the best match of a regex according to how close the indices are to the original indices.
        Indices should match perfectly for RoBERTa, which is not nessecarily the case for BERT tokens.

        Args:
            matches (List[Tuple[int, int]]): List of spans from regex matches, containing start and end indices
            tokens (list): List of tokens
            t_start (int): Index of the begin token of the entity
            t_end (int): Index of the last inside token of the entity

        Returns:
            Tuple[int, int]: Tuple containing the character indices of the original utterance
        """
        # If regex matches multiple times, choose best
        if self.pretrained_model_name != "roberta-base":
            warn("Model not supported in _find_best_match")
        num_chars = sum([len(t) for t in tokens[: t_end + 1]])
        return sorted(matches, key=lambda m: abs(m[1] - num_chars))[0]

    def format2original(
        self, intents: torch.Tensor, token_labels_matrix: torch.Tensor
    ) -> dict:
        """Format back to the original JSON format.

        Args:
            intents (torch.Tensor): Tensor of the predicted intent-ID's
            token_labels_matrix (torch.Tensor): Tensor of the tokens

        Returns:
            dict: results in the original JSON format
        """
        result = {}
        intents = [ID2INTENT[intent] for intent in intents]
        for i, (intent, token_labels, raw) in enumerate(
            zip(intents, token_labels_matrix, self.data_list)
        ):
            tokens = self.tokenizer.tokenize(raw["text"])[: self.sequence_length]
            entities = self._get_entities(token_labels)
            for j, (label, start, end) in enumerate(entities):
                # construct regex
                regex = " *".join(
                    [
                        token.replace(self.replace_char, ".").replace(
                            self.added_chars[self.pretrained_model_name], ""
                        )
                        for token in tokens[start : end + 1]
                    ]
                )
                # find indices
                matches = [m.span() for m in re.finditer(regex, raw["uncleaned"])]
                if len(matches) == 1:
                    match = matches[0]
                elif len(matches) == 0:
                    warn(f'Regex "{regex}" did not match in sample {i}')
                    continue
                else:
                    match = self._find_best_match(matches, tokens, start, end)
                # assgin character spans
                entities[j][1], entities[j][2] = match

            result[str(i)] = {
                "text": raw["text"],
                "intent": intent,
                "positions": {
                    ID2ENTITY[label][2:]: [start, end - 1]
                    for label, start, end in entities
                },
                "slots": {
                    ID2ENTITY[label][2:]: raw["uncleaned"][start:end]
                    for label, start, end in entities
                },
            }
        return result


if __name__ == "__main__":
    # End to end test for data conversion
    dm = DataModule(mode="fit", no_split=True)
    orig = {
        str(i): {key: value for key, value in data.items() if key != "uncleaned"}
        for i, data in enumerate(dm.data_list)
    }
    reconstruct = dm.format2original(
        dm.train.data["intent"].tolist(), dm.train.data["token_labels"].tolist()
    )
    errors = [
        (orig[str(i)], reconstruct[str(i)])
        for i in range(len(reconstruct))
        if orig[str(i)] != reconstruct[str(i)]
    ]
    inspect = input("Number of Errors: " + str(len(errors)) + "y/n")
    if inspect == "y":
        for i in range(len(reconstruct)):
            if orig[str(i)] != reconstruct[str(i)]:
                print(orig[str(i)])
                print(reconstruct[str(i)])
                print("\n")
                import pdb

                pdb.set_trace()
