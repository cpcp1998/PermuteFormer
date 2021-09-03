#!/bin/bash
fairseq-eval-lm data-bin/wikitext-103 --batch-size 2 --tokens-per-sample 512 --context-window 256 --path $@
