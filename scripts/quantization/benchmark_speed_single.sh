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

model_path=$1
save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_speedup/xxxxxxxx.csv"

echo $model_path

python src/benchmark_speed.py \
  --model_path $model_path \
  --save_file ${save_file} \
  # --model_type $model_type \
  # --pretrained

