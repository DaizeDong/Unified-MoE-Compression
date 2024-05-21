#!/usr/bin/bash

num_nodes=1
num_processes=2

#dataset="lima" # lima MetaMathQA
#prune_data_type="sft"

dataset="c4_train"
prune_data_type="pt"

n_calibration_samples=128
seq_len=2048

prune_method="sparsegpt" # wanda sparsegpt magnitude
sparsity_type="2:4"      # unstructured 4:8 2:4
sparsity_ratio=0.5

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
output_dir="########PATH_TO_SAVE_THE_RESULTS########"
prune_model_save_path=${output_dir}/checkpoint

accelerate launch \
  --config_file "config/accelerate/mixtral_deepspeed.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --sparsity_ratio ${sparsity_ratio} \
  --prune_method ${prune_method} \
  --sparsity_type ${sparsity_type} \
  --prune_model_save_path ${prune_model_save_path}
