import os
import json
import math
import argparse
import re
import codecs
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from models import MODEL_MAP

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
NUM_SPECIAL_TOKEN = 3


def format_num(n):
    f = '{0:.4g}'.format(n).replace('+0', '+').replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n

import tqdm
tqdm.tqdm.format_num = format_num


class ImdbBytesDataset(Dataset):
    def __init__(self, text, label):
        assert len(text) == len(label)
        self.text = [codecs.encode(i) for i in text]
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]


class TextClassificationModel(pl.LightningModule):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_size, 1)
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        pooler_output = self.encoder(input_ids, attention_mask)["pooler_output"]
        logits = self.decoder(pooler_output).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float))
        self.train_acc(F.sigmoid(logits), labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float))
        self.valid_acc(F.sigmoid(logits), labels)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=0.0)
        warmup = 4000
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            return 1 / math.sqrt(step / warmup)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
        }
        return [optimizer], [scheduler]


FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+).txt$")

def load_dataset(path):
    logger.info(f"Loading {path}")
    text = []
    label = []
    for label_id, label_name in ((0, "neg"), (1, "pos")):
        subpath = os.path.join(path, label_name)
        text_dict = {}
        for filename in os.listdir(subpath):
            match = FILENAME_PATTERN.match(filename)
            if not match:
                continue
            unique_id = int(match.group(1))
            score = int(match.group(2))
            assert unique_id not in text_dict
            filename = os.path.join(subpath, filename)
            with open(filename) as f:
                content = f.read()
            content = content.strip()
            assert "\n" not in content
            text_dict[unique_id] = content
        for i in range(len(text_dict)):
            text.append(text_dict[i])
            label.append(label_id)
    dataset = ImdbBytesDataset(text, label)

    text = dataset.text
    logger.info(f"{len(text)} seqs, max len {max(len(i) for i in text)}, avg len {sum(len(i) for i in text)/len(text)}")
    logger.info(f"#neg: {sum(1 for i in label if i == 0)}, #pos: {sum(1 for i in label if i == 1)}")
    return dataset



def load_datasets(args):
    train_path = os.path.join(args.data_dir, "train")
    test_path = os.path.join(args.data_dir, "test")

    train_dataset = load_dataset(train_path)
    test_dataset = load_dataset(test_path)

    train_dataset.text = [i[:args.max_length - 2] for i in train_dataset.text]
    test_dataset.text = [i[:args.max_length - 2] for i in test_dataset.text]

    return train_dataset, test_dataset


def load_model(args, vocab_size, max_position_embeddings):
    with open(args.model_config) as f:
        json_config = json.load(f)
    json_config["pad_token_id"] = PAD_TOKEN_ID
    json_config["max_position_embeddings"] = max_position_embeddings
    json_config["vocab_size"] = vocab_size
    config_type, model_type = MODEL_MAP[json_config["model_type"]]
    config = config_type.from_dict(json_config)
    model = model_type(config)
    return model, json_config["hidden_size"]


def collate_fn(examples):
    bsz = len(examples)

    input_ids = [[int(j) + NUM_SPECIAL_TOKEN for j in i[0]] for i in examples]
    labels = [i[1] for i in examples]

    input_ids = [[CLS_TOKEN_ID] + i + [SEP_TOKEN_ID] for i in input_ids]
    max_len = max(len(i) for i in input_ids)
    pad_input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, seq in enumerate(input_ids):
        pad_input_ids[i, :len(seq)] = torch.LongTensor(seq)
        attention_mask[i, :len(seq)] = 1
    labels = torch.LongTensor(labels)
    return pad_input_ids, attention_mask, labels


def stats_classifier(train_dataset, test_dataset):
    model = collections.defaultdict(collections.Counter)
    for i in range(len(train_dataset)):
        source, label = train_dataset[i]
        model[source[0]].update([label])
    model = {k: v.most_common(1)[0][0] for k, v in model.items()}

    correct = 0
    for i in range(len(test_dataset)):
        source, label = test_dataset[i]
        if source[0] in model and model[source[0]] == label:
            correct += 1
    return correct / len(test_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data/aclImdb")
    parser.add_argument("--model-config", default="configs/bert_prenorm_tc.json")
    parser.add_argument("--max-length", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=10)
    args = parser.parse_args()

    train_dataset, test_dataset = load_datasets(args)
    logger.info(f"Baseline on test: {stats_classifier(train_dataset, test_dataset)}")
    encoder, hidden_size = load_model(args, NUM_SPECIAL_TOKEN + 256, args.max_length)
    model = TextClassificationModel(encoder, hidden_size)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    trainer = pl.Trainer(gpus=8, accelerator="ddp", max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, test_loader)

if __name__ == "__main__":
    main()
