#!/usr/bin/bash

num_nodes=1
num_processes=1

#dataset="lima" # lima MetaMathQA
#data_type="sft"

dataset="c4_train"
data_type="pt"

n_compression_samples=128
seq_len=2048

compress_method="expert_drop"
expert_drop_method="global_pruning" # layerwise_pruning global_pruning
reverse_drop="False"                # False True
preserve_gate="False"               # False True
r=48                                # 64 60 56 48 44 40 36 32 28 24 20 16 12 8 4 0

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########" # also support quantized models
output_dir="########PATH_TO_SAVE_THE_RESULTS########"

output_dir="${output_dir}/DeepSeek-${compress_method}-${expert_drop_method}-r${r}-${dataset}${n_compression_samples}samples"
compressed_model_save_path=${output_dir}/checkpoint
use_fast_tokenizer="True"

accelerate launch \
  --config_file "config/accelerate/deepseek_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_compress.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --dataset ${dataset} \
  --split "train" \
  --data_type ${data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_compression_samples ${n_compression_samples} \
  --compress_method ${compress_method} \
  --expert_drop_method ${expert_drop_method} \
  --r ${r} \
  --reverse_drop ${reverse_drop} \
  --preserve_gate ${preserve_gate} \
  --compressed_model_save_path ${compressed_model_save_path}

accelerate launch \
  --config_file "config/accelerate/deepseek_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_compress.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --dataset ${dataset} \
  --split "train" \
  --data_type ${data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_compression_samples ${n_compression_samples} \
  --compress_method ${compress_method} \
  --expert_drop_method "post_dropping" \
  --r ${r} \
  --reverse_drop ${reverse_drop} \
  --preserve_gate ${preserve_gate} \
  --compressed_model_save_path ${compressed_model_save_path}
