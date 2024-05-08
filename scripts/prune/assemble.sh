#!/usr/bin/bash

#SBATCH --job-name=block
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_assemble/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_assemble/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=auto
# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=8 # should match with --gres
export OMP_NUM_THREADS=8
export LOGLEVEL=INFO

{
  # @Desc 此脚本用于获取一个指定区间且未被占用的随机端口号
  # @Author Hellxz <hellxz001@foxmail.com>

  function Listening { #判断当前端口是否被占用，没被占用返回0，反之1
    TCPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l)
    UDPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l)
    ((Listeningnum = TCPListeningnum + UDPListeningnum))
    if [ $Listeningnum == 0 ]; then
      echo "0"
    else
      echo "1"
    fi
  }

  function get_random_port { #得到随机端口
    PORT=0
    while [ $PORT == 0 ]; do
      temp_port=$(shuf -i $1-$2 -n1) #指定区间随机数
      if [ $(Listening $temp_port) == 0 ]; then
        PORT=$temp_port
      fi
    done
    echo "$PORT"
  }

  port=$(get_random_port 29500 29600) #任取一个未占用端口号
  echo "Port: $port"
}

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"
echo "Node list: $SLURM_JOB_NODELIS"

num_processes=$(expr ${num_nodes} \* ${num_gpu_per_node})
echo "Total Nodes: $num_nodes"
echo "Total GPUs: $num_processes"

#############################################  Block Drop  #############################################

# dataset="c4_train"
#dataset=alpaca-gpt4_de,wiki_demo,sharegpt4,dolly_15k_de,dolly_15k_de,c4_demo
#dataset=alpaca-gpt4_de,c4_valid
dataset=alpaca-gpt4_de
prune_data_type="pt"

n_calibration_samples=1024
seq_len=2048

prune_method="block_drop"
# block_drop_method="consecutive"
block_drop_method="discrete"
drop_n=4

similarity_cache_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/cache/Mixtral-block-${dataset}-${n_calibration_samples}-block-samples.pt"
model_name_or_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
folder_name="Mixtral-${prune_method}-${block_drop_method}-drop${drop_n}"
output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

echo ${folder_name}

source ~/anaconda3/bin/activate compression
cd /mnt/petrelfs/dongdaize.d/workspace/compression

srun accelerate launch \
  --config_file "config/accelerate/mixtral_deepspeed.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_ip ${head_node_ip} \
  --main_process_port ${port} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}

# Save the converted the model without DeepSpeed
block_drop_method="post_dropping"

srun accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_ip ${head_node_ip} \
  --main_process_port ${port} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}


#############################################  Expert Drop  #############################################

n_calibration_samples=128
seq_len=2048
r=7

prune_method="expert_drop"
expert_drop_method="layerwise_pruning"

model_name_or_path=$prune_model_save_path
previous_folder_name=$folder_name
folder_name="$folder_name-${prune_method}-${expert_drop_method}-r${r}"
output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/${folder_name}
prune_model_save_path=${output_dir}/checkpoint
similarity_cache_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/cache/Mixtral-layer-${dataset}-${n_calibration_samples}-layer-samples.pt"

echo ${folder_name}

conda deactivate
source activate compression
cd /mnt/petrelfs/dongdaize.d/workspace/compression

srun accelerate launch \
  --config_file "config/accelerate/mixtral_deepspeed.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_ip ${head_node_ip} \
  --main_process_port ${port} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --expert_drop_method ${expert_drop_method} \
  --r ${r} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}


expert_drop_method="post_dropping"
srun accelerate launch \
  --config_file "config/accelerate/mixtral_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_ip ${head_node_ip} \
  --main_process_port ${port} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --expert_drop_method ${expert_drop_method} \
  --r ${r} \
  --prune_model_save_path ${prune_model_save_path}

rm -rf ${previous_folder_name} # remove folders created by the last step

#############################################  Quantization  #############################################

model_name_or_path=$prune_model_save_path
quantized_model_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble
bits=4
seed=0
num_samples=128

calibration_template=mistral
calibration_template=default

quantization_method="GPTQ"
abbreviation=${model##*/}-$quantization_method-${bits}bits/checkpoint
previous_folder_name=$folder_name
output_dir=$quantized_model_dir/$folder_name/checkpoint
folder_name="$previous_folder_name-$quantization_method-${bits}bits"

if [ $quantization_method == "GPTQ" ]; then
  source activate awq
  cd /mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/gptq-main/zeroShot
  python gptq_auto.py \
  --pretrained_model_dir $model \
  --quantized_model_dir $$quantized_model_dir/$folder_name/checkpoint \
  --bits $bits --save_and_reload --desc_act \
  --seed $seed --num_samples $num_samples \
  --calibration-template $calibration_template
else
  source activate quant
  cd /mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/AutoAWQ
  python examples/quantize.py $model_path $$quantized_model_dir/$folder_name/checkpoint $bits
fi

rm -rf ${previous_folder_name} # remove folders created by the last step
