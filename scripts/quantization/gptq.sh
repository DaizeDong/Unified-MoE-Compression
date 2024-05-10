#!/usr/bin/bash

#SBATCH --job-name=gptq
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_quantization/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_quantization//%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# SBATCH --quotatype=spot
#SBATCH --quotatype=auto
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
source activate awq

cd /mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/gptq-main/zeroShot

model=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
model=${model_path##*/}
# model=/mnt/petrelfs/share_data/quxiaoye/models/Mistral-7B-v0.1
# model=/mnt/petrelfs/share_data/quxiaoye/models/Llama_2_13b_chat

quantized_model_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization
bits=4
seed=0
num_samples=128
# calibration_template=llama-2
calibration_template=mistral
calibration_template=default

abbreviation=${model##*/}-GPTQ-${bits}bits/checkpoint

python gptq_auto.py --pretrained_model_dir $model --quantized_model_dir $quantized_model_dir/$abbreviation/checkpoint \
                    --bits $bits --save_and_reload --desc_act \
                    --seed $seed --num_samples $num_samples \
                    --calibration-template $calibration_template
