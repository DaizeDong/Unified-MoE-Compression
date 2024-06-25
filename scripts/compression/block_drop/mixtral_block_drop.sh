#!/usr/bin/bash

num_nodes=1
num_processes=2

#dataset="lima" # lima MetaMathQA
#data_type="sft"

dataset="c4_train"
data_type="pt"

n_compression_samples=128
seq_len=2048

compress_method="block_drop"
block_drop_method="discrete" # discrete consecutive
drop_n=5

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########" # also support quantized models
output_dir="########PATH_TO_SAVE_THE_RESULTS########"
similarity_cache_file="########PATH_TO_SAVE_THE_CACHE########"

output_dir="${output_dir}/Mixtral-${compress_method}-${block_drop_method}-drop${drop_n}-${dataset}${n_compression_samples}samples"
similarity_cache_file="${similarity_cache_file}/Mixtral-${compress_method}-${dataset}-${n_compression_samples}samples.pt"
compressed_model_save_path=${output_dir}/checkpoint

accelerate launch \
  --config_file "config/accelerate/mixtral_deepspeed.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --data_type ${data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_compression_samples ${n_compression_samples} \
  --compress_method ${compress_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --compressed_model_save_path ${compressed_model_save_path}

accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --data_type ${data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_compression_samples ${n_compression_samples} \
  --compress_method ${compress_method} \
  --block_drop_method "post_dropping" \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --compressed_model_save_path ${compressed_model_save_path}
