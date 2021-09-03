import os
import json
import math
import argparse
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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


class PathfinderDataset(Dataset):
    def __init__(self, image, label):
        assert len(image) == len(label)
        self.image = image
        self.label = label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx], self.label[idx]


class ImageModel(pl.LightningModule):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_size, 1)
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, input_ids):
        pooler_output = self.encoder(input_ids)["pooler_output"]
        logits = self.decoder(pooler_output).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float))
        self.train_acc(F.sigmoid(logits), labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float))
        self.valid_acc(F.sigmoid(logits), labels)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=0.0)
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


FILENAME_PATTERN = re.compile(r"^(\d+).npy$")

def load_datasets(args):
    logger.info(f"Loading {args.data_dir}")

    metafiles = {}
    for metafile in os.listdir(os.path.join(args.data_dir, "metadata")):
        match = FILENAME_PATTERN.match(metafile)
        if not match:
            continue
        idx = int(match.group(1))
        assert idx not in metafiles
        metafiles[idx] = metafile
    assert all(i in metafiles for i in range(len(metafiles)))

    image = []
    label = []
    for idx in range(len(metafiles)):
        metafile = metafiles[idx]
        with open(os.path.join(args.data_dir, "metadata", metafile)) as f:
            for line in f:
                img_dir, img_name, _, label_, *_ = line.split()
                try:
                    img = Image.open(os.path.join(args.data_dir, img_dir, img_name))
                except:
                    logger.info(f"Cannot read {os.path.join(args.data_dir, img_dir, img_name)}")
                    continue
                assert img.width == img.height == args.size
                img = list(img.getdata())
                assert 0 <= min(img) and max(img) < 256
                label_ = int(label_)
                assert label_ in (0, 1)
                image.append(img)
                label.append(label_)
    logger.info(f"Loaded {len(image)} samples")

    split = [0, int(len(image) * 0.8), int(len(image) * 0.9), len(image)]
    train_dataset = PathfinderDataset(image[split[0]: split[1]], label[split[0]: split[1]])
    val_dataset = PathfinderDataset(image[split[1]: split[2]], label[split[1]: split[2]])
    test_dataset = PathfinderDataset(image[split[2]: split[3]], label[split[2]: split[3]])
    logger.info(f"train: {len(train_dataset)}, pos ratio: {sum(train_dataset.label)/len(train_dataset)}")
    logger.info(f"val: {len(val_dataset)}, pos ratio: {sum(val_dataset.label)/len(val_dataset)}")
    logger.info(f"test: {len(test_dataset)}, pos ratio: {sum(test_dataset.label)/len(test_dataset)}")
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

    input_ids = [[CLS_TOKEN_ID] + [NUM_SPECIAL_TOKEN + j for j in i] + [SEP_TOKEN_ID] for i in input_ids]
    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../data/lra_release/lra_release/pathfinder32/curv_contour_length_14")
    parser.add_argument("--model-config", default="configs/bert_prenorm_pathfinder.json")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=80)
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_datasets(args)
    encoder, hidden_size = load_model(args, NUM_SPECIAL_TOKEN + 256, args.size ** 2 + 2)
    model = ImageModel(encoder, hidden_size)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    trainer = pl.Trainer(gpus=8, accelerator="ddp", max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
