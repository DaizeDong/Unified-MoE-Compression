#!/usr/bin/bash

root_path="/mnt/petrelfs/dongdaize.d/workspace/compression/"

dataset="c4_train"
prune_data_type="pt"
n_calibration_samples=128
seq_len=2048
layer_drop_method="discrete"
layer_drop_norm="True"

#for drop_n in {1..27}; do
for drop_n in 15; do
  echo ${dataset} ${prune_data_type}
  echo ${n_calibration_samples} ${seq_len}
  echo ${layer_drop_method} ${drop_n} ${layer_drop_norm}
  sbatch ${root_path}/scripts/prune_auto/args_launch/deepseek_layer_drop_args_launch.sh \
    ${dataset} \
    ${prune_data_type} \
    ${n_calibration_samples} \
    ${seq_len} \
    ${layer_drop_method} \
    ${drop_n} \
    ${layer_drop_norm}
  sleep 1
done
