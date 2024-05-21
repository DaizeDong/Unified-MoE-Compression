#!/usr/bin/bash

root_path="/mnt/petrelfs/dongdaize.d/workspace/compression/"

dataset="c4_train"
prune_data_type="pt"
n_calibration_samples=128
seq_len=2048
expert_drop_method="layerwise_pruning" # layerwise_pruning global_pruning
reverse_drop="False"
preserve_gate="False"

for ((r = 64; r >= 0; r -= 4)); do
  echo ${dataset} ${prune_data_type}
  echo ${n_calibration_samples} ${seq_len}
  echo ${expert_drop_method} ${reverse_drop} ${preserve_gate} ${r}
  sbatch ${root_path}/scripts/prune_auto/args_launch/deepseek_expert_drop_args_launch.sh \
    ${dataset} \
    ${prune_data_type} \
    ${n_calibration_samples} \
    ${seq_len} \
    ${expert_drop_method} \
    ${reverse_drop} \
    ${preserve_gate} \
    ${r}
  sleep 1
done
