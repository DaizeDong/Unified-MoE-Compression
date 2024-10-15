#!/usr/bin/bash

# This script doesn't support finetuning quantized models.

num_nodes=1
num_processes=8

dataset="alpaca_gpt4_en"

total_batch_size=32
grad_accum=1
lr=8e-6
epochs=3.0

each_device_batch_size=$((${total_batch_size} / ${grad_accum} / ${num_processes} / ${num_nodes}))
echo "Training mistral model using ${num_processes} GPUs, $each_device_batch_size batch size per GPU, ${grad_accum} gradient accumulation steps"

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
output_dir="########PATH_TO_SAVE_THE_RESULTS########"

output_dir="${output_dir}/DeepSeek-${dataset}-lr${lr}-epoch${epochs}"
use_fast_tokenizer="True"

accelerate launch \
  --config_file "config/accelerate/deepseek_deepspeed_zero2.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_compress.py \
  --stage "sft" \
  --do_train \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --dataset ${dataset} \
  --template default \
  --finetuning_type full \
  --overwrite_output_dir \
  --output_dir ${output_dir} \
  --per_device_train_batch_size ${each_device_batch_size} \
  --gradient_accumulation_steps ${grad_accum} \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --save_strategy "no" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate ${lr} \
  --num_train_epochs ${epochs} \
  --plot_loss \
  --bf16 \
  --report_to tensorboard

# You should be careful of LoRA as it doesn't support MoE with so many experts (e.g. 64 for DeepSeek) well, which may be quite slow during training.
#lora_rank=8
#  --disable_gradient_checkpointing \
#  --finetuning_type lora \
#  --lora_rank ${lora_rank} \
#  --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
