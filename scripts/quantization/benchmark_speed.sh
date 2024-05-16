#!/usr/bin/bash

#SBATCH --job-name=benchmark_speedup
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_quantization/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_quantization/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=auto
# reserved spot auto

#####################################################################################################################
source ~/anaconda3/bin/activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression

model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
# model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mistral-7B-v0.1
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek

# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/deepseek-AWQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/deepseek-GPTQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mistral-7B-v0.1-AWQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mistral-7B-v0.1-GPTQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mixtral-8x7B-v0.1-AWQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mixtral-8x7B-v0.1-GPTQ-4bits/checkpoint
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/DeepSeek-expert_drop-global_pruning-r56-block_drop-discrete-drop4/checkpoint
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/Mixtral-block_drop-discrete-drop17/checkpoint

python src/benchmark_speed.py \
  --model_path $model_path \
  --pretrained \
