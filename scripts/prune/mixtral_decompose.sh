#!/usr/bin/bash
### bash scripts/prune/mixtral_decompose.sh ###

model_name_or_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
folder_name="Mixtral-decompose_moe"

output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

sbatch /mnt/petrelfs/dongdaize.d/workspace/compression/scripts/prune/mixtral_decompose_args_launch.sh \
  ${model_name_or_path} \
  ${output_dir} \
  ${prune_model_save_path}

##############################################################################
# 监测模型是否保存，循环直到文件存在且时间戳大于脚本开始时间
LISTEN_FILE_NAME=${prune_model_save_path}/tokenizer.model # 监测tokenizer，因为它是最后保存的
START_TIME=$(date +%s)

while true; do
  if [ -f ${LISTEN_FILE_NAME} ]; then # 文件存在
    FILE_MOD_TIME=$(date -r ${LISTEN_FILE_NAME} +%s)
    if [ "$FILE_MOD_TIME" -gt "$START_TIME" ]; then # 比较文件时间戳和开始时间戳
      echo "[$(date +%Y-%m-%d\ %H:%M:%S)] - ${LISTEN_FILE_NAME} has appeared and is newer than the initial checkpoint."
      break
    fi
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] - Waiting for ${LISTEN_FILE_NAME} to update..."
  else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] - Waiting for ${LISTEN_FILE_NAME} to appear..."
  fi
  sleep 60 # 每60秒检查一次
done

echo "${LISTEN_FILE_NAME} has appeared."
##############################################################################

output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_pt/${folder_name}
#dataset=alpaca-gpt4_de,wiki_demo,sharegpt4,dolly_15k_de,dolly_15k_de,c4_demo
#dataset=alpaca-gpt4_de,c4_valid
dataset=alpaca-gpt4_de

sbatch /mnt/petrelfs/dongdaize.d/workspace/compression/scripts/pt/mixtral_pt_eval_args_launch.sh \
  ${prune_model_save_path} \
  ${output_dir} \
  ${dataset}
