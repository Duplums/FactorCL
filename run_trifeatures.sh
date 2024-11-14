#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main_trifeatures.py \
          --root /home/bdufumier/results/align_or_not/FactorCL/bimodal_trifeatures \
          --run 0 \
          --biased true \
          --lr 1e-3