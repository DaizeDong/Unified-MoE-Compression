#!/usr/bin/bash

num_nodes=1
num_processes=1

#dataset="lima" # lima MetaMathQA
#prune_data_type="sft"

dataset="c4_train"
prune_data_type="pt"

n_calibration_samples=128
seq_len=2048

prune_method="block_drop"
block_drop_method="discrete" # consecutive discrete
drop_n=4
similarity_cache_file="########PATH_TO_SAVE_THE_CACHE########/DeepSeek-AWQ-block-${dataset}-${n_calibration_samples}samples.pt"

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT(SHOULD_BE_THE_QUANTIZED_AWQ_MODEL)#########"
output_dir="########PATH_TO_SAVE_THE_RESULTS########"
prune_model_save_path=${output_dir}/checkpoint
use_fast_tokenizer="True"
autoawq="True"   # True False
autogptq="False" # True False

mkdir ${prune_model_save_path}
cp ${model_name_or_path}/quantize_config.json ${prune_model_save_path}/quantize_config.json

accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --autoawq ${autoawq} \
  --autogptq ${autogptq} \
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
  --prune_model_save_path ${prune_model_save_path} \
  --similarity_cache_file ${similarity_cache_file}

accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --autoawq ${autoawq} \
  --autogptq ${autogptq} \
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
  --prune_model_save_path ${prune_model_save_path} \
  --similarity_cache_file ${similarity_cache_file}
