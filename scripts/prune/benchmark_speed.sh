#!/usr/bin/bash

#SBATCH --job-name=ben
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_prune/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_prune/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=auto
# reserved spot auto

#####################################################################################################################
source ~/anaconda3/bin/activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression

model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_speedup/mixtral_speed.csv"

# model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mistral-7B-v0.1
#save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_speedup/mistral_speed.csv"

#model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
#save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_speedup/deepseek_speed.csv"

python src/benchmark_speed.py \
  --model_path $model_path \
  --model_type "normal" \
  --save_file ${save_file} \
  --pretrained
