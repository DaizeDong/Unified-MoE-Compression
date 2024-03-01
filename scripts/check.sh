#!/usr/bin/bash

#####################################################################################################################
source ~/anaconda3/bin/activate compression
cd /mnt/petrelfs/dongdaize.d/workspace/compression

OMP_NUM_THREADS=8 srun --partition=MoE --job-name=temp --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --kill-on-bad-exit=1 \
  python src/llmtuner/train/prune/check.py
