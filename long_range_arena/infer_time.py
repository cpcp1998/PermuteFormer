import json
import argparse

import torch
import torch.utils.benchmark as benchmark

from models import MODEL_MAP

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
NUM_SPECIAL_TOKEN = 3


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


    input_ids = torch.LongTensor(input_ids)
    labels = torch.LongTensor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/bert_prenorm_pathfinder.json")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--vocab-size", type=int, default=259)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    encoder, hidden_size = load_model(args, args.vocab_size, args.max_length)
    dummy_input = torch.ones(args.batch_size, args.max_length, dtype=int)

    with torch.no_grad():
        for _ in range(2):
            encoder(dummy_input)
        timer = benchmark.Timer("encoder(dummy_input)", globals={"encoder": encoder, "dummy_input": dummy_input})
        print(timer.timeit(args.runs))

        encoder = encoder.cuda()
        dummy_input = dummy_input.cuda()
        for _ in range(2):
            encoder(dummy_input)
        timer = benchmark.Timer("encoder(dummy_input)", globals={"encoder": encoder, "dummy_input": dummy_input})
        print(timer.timeit(10 * args.runs))


if __name__ == "__main__":
    main()
