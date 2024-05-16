#!/usr/bin/bash

#SBATCH --job-name=bp_de
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_prune/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_prune/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
# SBATCH --quotatype=reserved
# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=1 # should match with --gres
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

##############################################################################
dataset="c4_train"
prune_data_type="pt"

n_calibration_samples=128
seq_len=2048

prune_method="expert_drop"
expert_drop_method="global_pruning" # layerwise_pruning global_pruning
reverse_drop="False"                # False True
preserve_gate="True"                # False True

for ((r = 60; r >= 0; r -= 4)); do
  model_name_or_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
  folder_name="DeepSeek-${prune_method}-${expert_drop_method}-r${r}"
  if [ ${reverse_drop} = "True" ]; then
    folder_name="${folder_name}-Reversed"
  fi
  if [ ${preserve_gate} = "True" ]; then
    folder_name="${folder_name}-DyGate"
  fi
  use_fast_tokenizer="True" # 🔍 necessary for DeepSeek
  echo ${folder_name}

  output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/${folder_name}
  prune_model_save_path=${output_dir}/checkpoint

  source ~/anaconda3/bin/activate compression
  cd /mnt/petrelfs/dongdaize.d/workspace/compression

  srun accelerate launch \
    --config_file "config/accelerate/deepseek_normal.yaml" \
    --num_processes ${num_processes} \
    --num_machines ${num_nodes} \
    --main_process_ip ${head_node_ip} \
    --main_process_port ${port} \
    src/train_bash.py \
    --stage prune \
    --model_name_or_path ${model_name_or_path} \
    --use_fast_tokenizer ${use_fast_tokenizer} \
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
    --reverse_drop ${reverse_drop} \
    --preserve_gate ${preserve_gate} \
    --prune_model_save_path ${prune_model_save_path}

  srun accelerate launch \
    --config_file "config/accelerate/deepseek_normal.yaml" \
    --num_processes ${num_processes} \
    --num_machines ${num_nodes} \
    --main_process_ip ${head_node_ip} \
    --main_process_port ${port} \
    src/train_bash.py \
    --stage prune \
    --model_name_or_path ${model_name_or_path} \
    --use_fast_tokenizer ${use_fast_tokenizer} \
    --dataset ${dataset} \
    --split "train" \
    --prune_data_type ${prune_data_type} \
    --cutoff_len ${seq_len} \
    --output_dir ${output_dir} \
    --logging_steps 10 \
    --bf16 \
    --n_calibration_samples ${n_calibration_samples} \
    --prune_method ${prune_method} \
    --expert_drop_method "post_dropping" \
    --r ${r} \
    --reverse_drop ${reverse_drop} \
    --preserve_gate ${preserve_gate} \
    --prune_model_save_path ${prune_model_save_path}

  #####################################################################################################################
  source ~/anaconda3/bin/activate awq
  cd /mnt/petrelfs/dongdaize.d/workspace/compression

  save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_speedup/${folder_name}_speed.csv"

  python src/benchmark_speed.py \
    --model_path ${prune_model_save_path} \
    --model_type "normal" \
    --save_file ${save_file} \
    --pretrained

  rm -rf ${output_dir}
done
