#!/usr/bin/bash

##############################################################################

model_path=mistralai/Mixtral-8x7B-v0.1
model=${model_path##*/}

bits=4
seed=0
num_samples=16
calibration_template=default

quantized_model_dir=./results_quantization
abbreviation=${model##*/}-GPTQ-${bits}bits

python gptq_auto.py --pretrained_model_dir $model_path --quantized_model_dir $quantized_model_dir/$abbreviation \
  --bits $bits --save_and_reload --desc_act \
  --seed $seed --num_samples $num_samples \
  --calibration-template $calibration_template \
  --trust_remote_code \
  --use_triton \
