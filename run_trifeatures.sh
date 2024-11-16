#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main_trifeatures.py \
          --root /home/bdufumier/results/align_or_not/FactorCL/bimodal_trifeatures \
          --root_dataset /fastdata/trifeatures_3combi \
          --run 0 \
          --biased false \
          --lr 1e-4