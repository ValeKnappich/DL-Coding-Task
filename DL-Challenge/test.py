from model import IntentAndEntityModel
from data import DataModule
from config import *

import torch
import torch.nn.functional as F
import json


def main():
    dm = DataModule(mode="test", no_split=True)
    model = IntentAndEntityModel.load_from_checkpoint(config.ckpt_path)

    results = {}
    for batch in dm.test_dataloader():
        intent_logits, ner_logits = model(batch["input_ids"], batch["attention_mask"])
        intents = torch.argmax(F.softmax(intent_logits, dim=1), dim=1)
        token_labels = torch.argmax(
            F.softmax(
                ner_logits.view(ner_logits.shape[0] * ner_logits.shape[1], -1), dim=1
            ),
            dim=1,
        ).view(ner_logits.shape[0], ner_logits.shape[1])
        result = dm.format2original(intents.tolist(), token_labels.tolist())
        results.update(result)

    json.dump(results, open(config.dev_out_path, "w"))


if __name__ == "__main__":
    main()