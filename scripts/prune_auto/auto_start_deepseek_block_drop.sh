#!/usr/bin/bash

root_path="/mnt/petrelfs/dongdaize.d/workspace/compression/"

dataset="c4_train"
prune_data_type="pt"
n_calibration_samples=128
seq_len=2048
block_drop_method="discrete"

#for drop_n in {1..12}; do
#for drop_n in {13..27}; do
for drop_n in {1..27}; do
  echo ${dataset} ${prune_data_type}
  echo ${n_calibration_samples} ${seq_len}
  echo ${block_drop_method} ${drop_n}
  sbatch ${root_path}/scripts/prune_auto/args_launch/deepseek_block_drop_args_launch.sh \
    ${dataset} \
    ${prune_data_type} \
    ${n_calibration_samples} \
    ${seq_len} \
    ${block_drop_method} \
    ${drop_n}
  sleep 1
done
