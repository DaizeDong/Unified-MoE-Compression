#!/usr/bin/bash

num_nodes=1
num_processes=2

#dataset="lima" # lima MetaMathQA
#data_type="sft"

dataset="c4_train"
data_type="pt"

n_compression_samples=128
seq_len=2048

compress_method="layer_drop"
layer_drop_method="discrete"
drop_n=8
layer_drop_norm="True" # True False

model_name_or_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########" # also support quantized models
output_dir="########PATH_TO_SAVE_THE_RESULTS########"
similarity_cache_file="########PATH_TO_SAVE_THE_CACHE########"

if [ ${layer_drop_norm} = "True" ]; then
  output_dir="${output_dir}/Mixtral-${compress_method}-${layer_drop_method}-drop${drop_n}-${dataset}${n_compression_samples}samples"
  similarity_cache_file="${similarity_cache_file}/Mixtral-${compress_method}-${dataset}-${n_compression_samples}samples.pt"
else
  output_dir="${output_dir}/Mixtral-${compress_method}-${layer_drop_method}-drop${drop_n}-${dataset}${n_compression_samples}samples-NoNorm"
  similarity_cache_file="${similarity_cache_file}/Mixtral-${compress_method}-${dataset}-${n_compression_samples}samples-NoNorm.pt"
fi
compressed_model_save_path=${output_dir}/checkpoint

accelerate launch \
  --config_file "config/accelerate/mixtral_deepspeed_zero3.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_compress.py \
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
  --layer_drop_method ${layer_drop_method} \
  --drop_n ${drop_n} \
  --layer_drop_norm ${layer_drop_norm} \
  --similarity_cache_file ${similarity_cache_file} \
  --compressed_model_save_path ${compressed_model_save_path}

accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  src/run_compress.py \
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
  --layer_drop_method "post_dropping" \
  --drop_n ${drop_n} \
  --layer_drop_norm ${layer_drop_norm} \
  --similarity_cache_file ${similarity_cache_file} \
  --compressed_model_save_path ${compressed_model_save_path}
