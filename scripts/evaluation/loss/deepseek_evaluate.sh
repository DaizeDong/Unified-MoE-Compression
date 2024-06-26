#!/usr/bin/bash

num_nodes=1
num_processes=1

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
output_dir="########PATH_TO_SAVE_THE_RESULTS########"
use_fast_tokenizer="True"

dataset="c4_valid"

accelerate launch \
  --config_file "config/accelerate/deepseek_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_prune.py \
  --stage pt \
  --do_eval \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --dataset ${dataset} \
  --finetuning_type full \
  --output_dir ${output_dir} \
  --per_device_train_batch_size 4 \
  --logging_steps 10 \
  --plot_loss \
  --bf16
