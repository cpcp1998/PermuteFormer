import os
import json
import math
import argparse
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


class ListopsDataset(Dataset):
    def __init__(self, source, target, vocab, label):
        assert len(source) == len(target)
        self.source = source
        self.target = target
        self.vocab = vocab
        self.label = label

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


class ListopsModel(pl.LightningModule):
    def __init__(self, encoder, hidden_size, num_label):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_size, num_label)
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        pooler_output = self.encoder(input_ids, attention_mask)["pooler_output"]
        logits = self.decoder(pooler_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        self.train_acc(logits.argmax(dim=-1), labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        self.valid_acc(logits.argmax(dim=-1), labels)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0.0)
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
    with open(path) as f:
        tsv = [l.strip().split("\t") for l in f]
    assert tsv[0] == ["Source", "Target"]
    tsv = tsv[1:]
    assert all(len(i) == 2 for i in tsv)
    source = [i[0].split() for i in tsv]
    target = [int(i[1]) for i in tsv]

    # strip parenthesis
    source = [[j for j in i if j not in ("(", ")")] for i in source]

    # build vocab
    vocab = sorted(set(j for i in source for j in i))
    label = sorted(set(target))

    # tokenize
    tokenizer = {v: i + NUM_SPECIAL_TOKEN for i, v in enumerate(vocab)}
    source = [[tokenizer[j] for j in i] for i in source]

    logger.info(f"{len(source)} seqs, max len {max(len(i) for i in source)}, avg len {sum(len(i) for i in source) / len(source)}")

    dataset = ListopsDataset(source, target, vocab, label)
    return dataset


def load_datasets(args):
    train_path = os.path.join(args.data_dir, args.train_path)
    val_path = os.path.join(args.data_dir, args.val_path)
    test_path = os.path.join(args.data_dir, args.test_path)

    train_dataset = load_dataset(train_path)
    val_dataset = load_dataset(val_path)
    test_dataset = load_dataset(test_path)
    assert train_dataset.vocab == val_dataset.vocab == test_dataset.vocab
    assert train_dataset.label == val_dataset.label == test_dataset.label
    logger.info(f"#vocab {len(train_dataset.vocab)}, #label {len(train_dataset.label)}")

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

    input_ids = [i[0] for i in examples]
    labels = [i[1] for i in examples]

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
    parser.add_argument("--data-dir", default="../../data/lra_release/lra_release/listops-1000")
    parser.add_argument("--train-path", default="basic_train.tsv")
    parser.add_argument("--val-path", default="basic_val.tsv")
    parser.add_argument("--test-path", default="basic_test.tsv")
    parser.add_argument("--model-config", default="configs/bert_prenorm_listops.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=5)
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_datasets(args)
    logger.info(f"Baseline on val: {stats_classifier(train_dataset, val_dataset)}")
    logger.info(f"Baseline on test: {stats_classifier(train_dataset, test_dataset)}")
    vocab_size = NUM_SPECIAL_TOKEN + len(train_dataset.vocab)
    max_position_embeddings = max(
            max(len(i[0]) for i in train_dataset),
            max(len(i[0]) for i in val_dataset),
            max(len(i[0]) for i in test_dataset),
    )
    num_label = len(train_dataset.label)
    encoder, hidden_size = load_model(args, vocab_size, max_position_embeddings)
    model = ListopsModel(encoder, hidden_size, num_label)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    trainer = pl.Trainer(gpus=8, accelerator="ddp", max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
