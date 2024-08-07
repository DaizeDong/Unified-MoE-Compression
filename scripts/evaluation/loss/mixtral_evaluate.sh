#!/usr/bin/bash

num_nodes=1
num_processes=2

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
output_dir="########PATH_TO_SAVE_THE_RESULTS########"

dataset="c4_valid"
#dataset="c4_valid_full" # please download the full validation data before using this

accelerate launch \
  --config_file "config/accelerate/mixtral_deepspeed_zero3.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_compress.py \
  --stage pt \
  --do_eval \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --finetuning_type full \
  --output_dir ${output_dir} \
  --per_device_train_batch_size 4 \
  --logging_steps 10 \
  --plot_loss \
  --bf16
