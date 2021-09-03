import os
import json
import math
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

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


class ImageModel(pl.LightningModule):
    def __init__(self, encoder, hidden_size, num_label):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_size, num_label)
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, input_ids):
        pooler_output = self.encoder(input_ids)["pooler_output"]
        logits = self.decoder(pooler_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = F.cross_entropy(logits, labels)
        self.train_acc(logits.argmax(dim=-1), labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = F.cross_entropy(logits, labels)
        self.valid_acc(logits.argmax(dim=-1), labels)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=0.0)
        warmup = 200
        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            return 1 / math.sqrt(step / warmup)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def load_datasets(args):
    train_dataset = torchvision.datasets.CIFAR10(args.data_dir, download=True, train=True)
    test_dataset = torchvision.datasets.CIFAR10(args.data_dir, download=True, train=False)

    for d in (train_dataset, test_dataset):
        for i, l in d:
            assert i.width == i.height == args.size
            assert 0 <= l < args.num_label
        assert len(set(l for i, l in d)) == args.num_label

    logger.info(f"#train: {len(train_dataset)}, #test: {len(test_dataset)}")
    logger.info(f"image size: {train_dataset[0][0].size}, #labels: {args.num_label}")
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

   input_ids = [list(i[0].convert("L").getdata()) for i in examples]
   labels = [i[1] for i in examples]

   input_ids = [[CLS_TOKEN_ID] + [NUM_SPECIAL_TOKEN + j for j in i] + [SEP_TOKEN_ID] for i in input_ids]
   input_ids = torch.LongTensor(input_ids)
   labels = torch.LongTensor(labels)
   return input_ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data")
    parser.add_argument("--model-config", default="configs/bert_prenorm_image.json")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num-label", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()

    train_dataset, test_dataset = load_datasets(args)
    encoder, hidden_size = load_model(args, NUM_SPECIAL_TOKEN + 256, args.size ** 2 + 2)
    model = ImageModel(encoder, hidden_size, args.num_label)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    trainer = pl.Trainer(gpus=8, accelerator="ddp", max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
