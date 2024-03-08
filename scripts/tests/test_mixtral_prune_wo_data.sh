#!/usr/bin/bash

#SBATCH --job-name=ttt
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_prune/test/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_prune/test/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=1 # should match with --gres
export OMP_NUM_THREADS=8
export LOGLEVEL=INFO

# FOR DEBUG USAGE 加上这些才能正确打印报错位置(正式运行注释掉，会减速)
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_SHOW_CPP_STACKTRACES=1
#export CUDA_LAUNCH_BLOCKING=1

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

#######################################
model_name_or_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
n_calibration_samples=32
seq_len=512

sparsity_ratio=0.5
prune_method="magnitude" # magnitude
config_file="config/accelerate/mixtral_normal.yaml"
sparsity_type="unstructured"
output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/test-Mixtral-${prune_method}-${sparsity_type}-${sparsity_ratio}
prune_model_save_path=${output_dir}/checkpoint

source ~/anaconda3/bin/activate compression
cd /mnt/petrelfs/dongdaize.d/workspace/compression

# srun torchrun \
#   --nnodes ${num_nodes} \
#   --nproc_per_node ${num_gpu_per_node} \
#   --node_rank $SLURM_NODEID \
#   --rdzv_id $RANDOM \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $head_node:29518 \
#   src/evaluate_sparse.py \
#   --model_name_or_path $model_name_or_path \
#   --template vanilla \
#   --finetuning_type full \
#   --task $task \
#   --split test \
#   --lang en \
#   --plot_loss \
#   --n_shot 0 \
#   --batch_size 4 \
#   --stage prune \

srun accelerate launch \
  --config_file ${config_file} \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_ip ${head_node_ip} \
  --main_process_port ${port} \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset "lima" \
  --prune_data_type "sft" \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --sparsity_ratio ${sparsity_ratio} \
  --prune_method ${prune_method} \
  --sparsity_type ${sparsity_type} \
  --prune_model_save_path ${prune_model_save_path}
