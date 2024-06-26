#!/usr/bin/bash

num_nodes=1
num_processes=1

#dataset="lima" # lima MetaMathQA
#data_type="sft"

dataset="c4_train"
data_type="pt"

n_compression_samples=128
seq_len=2048

compress_method="sparsegpt" # wanda sparsegpt magnitude
sparsity_type="2:4"         # unstructured 4:8 2:4
sparsity_ratio=0.5
exclude_prune_module_name="shared_experts"

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
output_dir="########PATH_TO_SAVE_THE_RESULTS########"

output_dir="${output_dir}/DeepSeek-${compress_method}-${sparsity_ratio}-${sparsity_type}-${dataset}${n_compression_samples}samples-NoShared"
compressed_model_save_path=${output_dir}/checkpoint
use_fast_tokenizer="True"

accelerate launch \
  --config_file "config/accelerate/deepseek_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_prune.py \
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
  --sparsity_ratio ${sparsity_ratio} \
  --compress_method ${compress_method} \
  --sparsity_type ${sparsity_type} \
  --exclude_prune_module_name "${exclude_prune_module_name}" \
  --compressed_model_save_path ${compressed_model_save_path}
