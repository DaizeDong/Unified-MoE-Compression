#!/usr/bin/bash

save_file="./results_speedup/xxxxxxxx.csv"
model_path="mistralai/Mixtral-8x7B-v0.1"
model_type="normal"
echo $model_path

python src/benchmark_speed.py \
  --model_path $model_path \
  --save_file ${save_file} \
  # --model_type $model_type \
  # --pretrained

