#!/usr/bin/bash

num_nodes=1
num_processes=2

#dataset="lima" # lima MetaMathQA
#prune_data_type="sft"

dataset="c4_train"
prune_data_type="pt"

n_calibration_samples=128
seq_len=2048

prune_method="block_drop"
block_drop_method="discrete" # consecutive discrete
drop_n=5
similarity_cache_file="########PATH_TO_SAVE_THE_CACHE########/Mixtral-block-${dataset}-${n_calibration_samples}samples.pt"

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
  --prune_method ${prune_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}

accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
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
  --prune_method ${prune_method} \
  --block_drop_method "post_dropping" \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}
