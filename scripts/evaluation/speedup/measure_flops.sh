#!/usr/bin/bash

model_path="########PATH_TO_HUGGING_FACE_CHECKPOINT########" # shouldn't be quantized models
save_file="########PATH_TO_SAVE_THE_RESULTS########/flops.txt"

batch_size=1
seq_len=2048
device="cuda" # cpu cuda

python src/measure_flops.py \
  --model_name_or_path ${model_path} \
  --save_file ${save_file} \
  --batch_size ${batch_size} \
  --seq_len ${seq_len} \
  --device ${device}
