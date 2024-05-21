#!/usr/bin/bash

##############################################################################

model_path=mistralai/Mixtral-8x7B-v0.1
model=${model_path##*/}
bits=4

quant_path=./results_quantization/$model-AWQ-${bits}bits-new/checkpoint

cd /src/llmtuner/train/quantization/AutoAWQ
python examples/quantize.py $model_path $quant_path $bits