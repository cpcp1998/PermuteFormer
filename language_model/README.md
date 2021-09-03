# PermuteFormer on WikiText-103

## Dependency
- `fairseq` [link](https://github.com/pytorch/fairseq)
- `pytorch-fast-transformers` [link](https://github.com/idiap/fast-transformers)

## Data Preparation

Please follow instructions on [fairseq](https://github.com/pytorch/fairseq/tree/master/examples/language_model#training-a-transformer-language-model-with-the-cli-tools).

## Training
Execute scripts started with `run_`.

## Evaluation
To evalutate Transformer, run
```
./eval.sh /path/to/checkpoint
```

To evalutate PermuteFormer, run
```
./eval.sh /path/to/checkpoint --user-dir permute
```
