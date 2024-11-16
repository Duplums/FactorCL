#!/bin/bash

TOKENIZERS_PARALLELISM=0 CUDA_VISIBLE_DEVICES=1 python3 main_hateful_memes.py \
          --root /home/bdufumier/results/align_or_not/FactorCL/hateful_memes \
          --root_dataset /fastdata/hateful_memes \
          --run 0 \
          --lr 1e-4 &> hateful_memes_factorCL.log
