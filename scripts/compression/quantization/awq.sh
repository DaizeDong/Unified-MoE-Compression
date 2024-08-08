#!/usr/bin/bash

model_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
quant_path="########PATH_TO_SAVE_THE_QUANTIZED_MODEL########"
bits=4
n_samples=512
seqlen=2048

python AutoAWQ/quantize.py \
  $model_path \
  $quant_path \
  $bits \
  $n_samples \
  $seqlen