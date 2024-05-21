#!/usr/bin/bash

#SBATCH --job-name=flops
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_benchmark/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_benchmark/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
# reserved spot auto

#####################################################################################################################
source ~/anaconda3/bin/activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression

root_path="/mnt/petrelfs/dongdaize.d/workspace/compression"
batch_size=1
seq_len=2048
device="cpu"
#device="cuda"

model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_flops/batch${batch_size}_seq${seq_len}/deepseek_flops.json"
echo ${model_path}
python src/benchmark_flops.py \
  --model_name_or_path ${model_path} \
  --save_file ${save_file} \
  --batch_size ${batch_size} \
  --seq_len ${seq_len} \
  --device ${device}

model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/mixtral
save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_flops/batch${batch_size}_seq${seq_len}/mixtral_flops.json"
echo ${model_path}
python src/benchmark_flops.py \
  --model_name_or_path ${model_path} \
  --save_file ${save_file} \
  --batch_size ${batch_size} \
  --seq_len ${seq_len} \
  --device ${device}

folder_name_list=(
  "DeepSeek-expert_drop-global_pruning-r48"
  "DeepSeek-layer_drop-discrete-drop4"
  "DeepSeek-block_drop-discrete-drop4"
  "Mixtral-expert_drop-global_pruning-r6"
  "Mixtral-layer_drop-discrete-drop8"
  "Mixtral-block_drop-discrete-drop5"
)

for folder_name in "${folder_name_list[@]}"; do
  model_path="${root_path}/results_prune/${folder_name}/checkpoint"
  save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_flops/batch${batch_size}_seq${seq_len}/${folder_name}_flops.json"
  echo ${model_path}
  python src/benchmark_flops.py \
    --model_name_or_path ${model_path} \
    --save_file ${save_file} \
    --batch_size ${batch_size} \
    --seq_len ${seq_len} \
    --device ${device}
done

#for drop_n in {1..27}; do
#  folder_name="DeepSeek-block_drop-discrete-drop${drop_n}"
#  model_path="${root_path}/results_prune/${folder_name}/checkpoint"
#  save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_flops/batch${batch_size}_seq${seq_len}/${folder_name}_flops.json"
#  echo ${model_path}
#  python src/benchmark_flops.py \
#    --model_name_or_path ${model_path} \
#    --save_file ${save_file} \
#    --batch_size ${batch_size} \
#    --seq_len ${seq_len} \
#    --device ${device}
#done
