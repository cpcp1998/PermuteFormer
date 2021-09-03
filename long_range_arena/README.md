# PermuteFormer on Long-Range Arena

This repo contains code to train PermuteFormer on Long-Range Arena.

Please download dataset from [Long-Range Arena](https://github.com/google-research/long-range-arena) and extract it to `../../data`.
(Or anywhere you like, and specify it through command-line option `--data-dir`.)

To train PermuteFormer on Text Classficiation for example, run
```
python text_classification.py --model-config configs/performer_relative_tc.json
```

Config files are in `configs/`.
These started with `bert_prenorm` are for Transformer.
These started with `performer_prenorm` are for Performer.
These started with `performer_relative` are for PermuteFormer without 2D position encoding.
These started with `performer_2d_relative` are for PermuteFormer with 2D position encoding.
