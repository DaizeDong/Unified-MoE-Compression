#!/usr/bin/bash

model_path="########PATH_TO_HUGGING_FACE_CHECKPOINT########" # also support quantized models
save_path="########PATH_TO_SAVE_THE_RESULTS########"

use_fast_tokenizer=True # True for DeepSeek, False for Mixtral
autoawq=False           # for AWQ
autogptq=False          # for GPTQ

if [ $autoawq == True ] || [ $autogptq == True ]; then
  trust_remote_code=False
else
  trust_remote_code=True
fi

####################################################################
task_name_list=("arc_challenge" "boolq" "hellaswag" "mmlu" "openbookqa" "piqa" "rte" "winogrande")
num_fewshot_list=(0 0 0 0 0 0 0 0)
batch_size="auto"
max_length=4096

if [ ! -d ${model_path} ]; then
  echo "Model path \"${model_path}\" not exists. Do not apply the task."
else
  for ((i = 0; i < ${#task_name_list[@]}; i++)); do
    task_name=${task_name_list[i]}
    num_fewshot=${num_fewshot_list[i]}
    echo "${task_name}: ${num_fewshot} shot"

    this_save_path="${save_path}/${task_name}-${num_fewshot}shot"
    mkdir -p ${this_save_path}

    lm_eval \
      --model hf \
      --model_args pretrained=${model_path},dtype="bfloat16",parallelize=True,trust_remote_code=${trust_remote_code},use_fast_tokenizer=${use_fast_tokenizer},max_length=${max_length},autoawq=${autoawq},autogptq=${autogptq} \
      --tasks ${task_name} \
      --num_fewshot ${num_fewshot} \
      --batch_size ${batch_size} \
      --output_path ${this_save_path}
  done
fi
