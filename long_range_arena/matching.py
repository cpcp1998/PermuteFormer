import os
import json
import math
import argparse

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


class MatchingDataset(Dataset):
    def __init__(self, text1, text2, label):
        assert len(text1) == len(text2) == len(label)
        self.text1 = text1
        self.text2 = text2
        self.label = label

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        return self.text1[idx], self.text2[idx], self.label[idx]

    def prune(self, max_length):
        self.text1 = [i[:max_length] for i in self.text1]
        self.text2 = [i[:max_length] for i in self.text2]


class MatchingModel(pl.LightningModule):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1),
        )
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):
        pooler_output1 = self.encoder(input_ids1, attention_mask1)["pooler_output"]
        pooler_output2 = self.encoder(input_ids2, attention_mask2)["pooler_output"]
        logits = self.decoder(torch.cat((pooler_output1, pooler_output2), dim=-1)).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids1, input_ids2, attention_mask1, attention_mask2, labels = batch
        logits = self(input_ids1, input_ids2, attention_mask1, attention_mask2)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float))
        self.train_acc(F.sigmoid(logits), labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids1, input_ids2, attention_mask1, attention_mask2, labels = batch
        logits = self(input_ids1, input_ids2, attention_mask1, attention_mask2)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float))
        self.valid_acc(F.sigmoid(logits), labels)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=0.0)
        warmup = 1000
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            return 1 / math.sqrt(step / warmup)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def load_dataset(path):
    logger.info(f"Loading {path}")
    text1 = []
    text2 = []
    label = []
    for line in open(path):
        label_, _, _, text1_, text2_ = line.split("\t")
        label_ = eval(label_)
        text1_ = eval(text1_)
        text2_ = eval(text2_)
        assert type(label_) is float
        assert type(text1_) is bytes
        assert type(text2_) is bytes
        label_ = int(label_)
        assert label_ in (0, 1)
        text1.append(text1_)
        text2.append(text2_)
        label.append(label_)
    dataset = MatchingDataset(text1, text2, label)

    logger.info(f"{len(text1)} pairs")
    logger.info(f"max len {max(len(i)+len(j) for i, j in zip(text1, text2))}")
    logger.info(f"avg len {sum(len(i)+len(j) for i, j in zip(text1, text2))/len(text1)}")
    logger.info(f"neg ratio: {sum(1 for i in label if i == 0)/len(label)}, pos ratio: {sum(1 for i in label if i == 1)/len(label)}")
    return dataset


def load_datasets(args):
    train_path = os.path.join(args.data_dir, "new_aan_pairs.train.tsv")
    val_path = os.path.join(args.data_dir, "new_aan_pairs.eval.tsv")
    test_path = os.path.join(args.data_dir, "new_aan_pairs.test.tsv")

    train_dataset = load_dataset(train_path)
    val_dataset = load_dataset(val_path)
    test_dataset = load_dataset(test_path)

    train_dataset.prune(args.max_length - 2)
    val_dataset.prune(args.max_length - 2)
    test_dataset.prune(args.max_length - 2)

    return train_dataset, val_dataset, test_dataset


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

    input_ids1 = [[int(j) + NUM_SPECIAL_TOKEN for j in i[0]] for i in examples]
    input_ids2 = [[int(j) + NUM_SPECIAL_TOKEN for j in i[1]] for i in examples]
    labels = [i[2] for i in examples]

    input_ids1 = [[CLS_TOKEN_ID] + i + [SEP_TOKEN_ID] for i in input_ids1]
    max_len1 = max(len(i) for i in input_ids1)
    pad_input_ids1 = torch.zeros((bsz, max_len1), dtype=torch.long)
    attention_mask1 = torch.zeros((bsz, max_len1), dtype=torch.long)
    for i, seq in enumerate(input_ids1):
        pad_input_ids1[i, :len(seq)] = torch.LongTensor(seq)
        attention_mask1[i, :len(seq)] = 1

    input_ids2 = [[CLS_TOKEN_ID] + i + [SEP_TOKEN_ID] for i in input_ids2]
    max_len2 = max(len(i) for i in input_ids2)
    pad_input_ids2 = torch.zeros((bsz, max_len2), dtype=torch.long)
    attention_mask2 = torch.zeros((bsz, max_len2), dtype=torch.long)
    for i, seq in enumerate(input_ids2):
        pad_input_ids2[i, :len(seq)] = torch.LongTensor(seq)
        attention_mask2[i, :len(seq)] = 1

    labels = torch.LongTensor(labels)

    return pad_input_ids1, pad_input_ids2, attention_mask1, attention_mask2, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data/lra_release/lra_release/tsv_data")
    parser.add_argument("--model-config", default="configs/bert_prenorm_matching.json")
    parser.add_argument("--max-length", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=10)
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_datasets(args)
    encoder, hidden_size = load_model(args, NUM_SPECIAL_TOKEN + 256, args.max_length)
    model = MatchingModel(encoder, hidden_size)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    trainer = pl.Trainer(gpus=8, accelerator="ddp", max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
