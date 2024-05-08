#!/usr/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_pt/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_pt/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=auto

# SBATCH --quotatype=reserved
# SBATCH --quotatype=spot
# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=4 # should match with --gres
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
#model_name_or_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
#output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_pt/DeepSeek

folder_name="DeepSeek-wanda-c4_train-2:4-0.5-1024-NoAttn"
model_name_or_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/${folder_name}/checkpoint
output_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_pt/${folder_name}
use_fast_tokenizer="True"

#dataset=alpaca-gpt4_de,wiki_demo,sharegpt4,dolly_15k_de,dolly_15k_de,c4_demo
dataset=alpaca-gpt4_de
#dataset=c4_valid

source ~/anaconda3/bin/activate compression
cd /mnt/petrelfs/dongdaize.d/workspace/compression

srun accelerate launch \
  --config_file "config/accelerate/deepseek_normal.yaml" \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_ip ${head_node_ip} \
  --main_process_port ${port} \
  src/train_bash.py \
  --stage pt \
  --do_eval \
  --model_name_or_path ${model_name_or_path} \
  --use_fast_tokenizer ${use_fast_tokenizer} \
  --dataset ${dataset} \
  --finetuning_type full \
  --output_dir ${output_dir} \
  --per_device_train_batch_size 4 \
  --logging_steps 10 \
  --plot_loss \
  --bf16

#  --print_param_status \
