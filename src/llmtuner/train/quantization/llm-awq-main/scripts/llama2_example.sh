#!/usr/bin/bash

#SBATCH --job-name=AWQ
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/llm-awq-main/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/llm-awq-main/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=spot
# SBATCH --quotatype=auto
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


MODEL=llama-2-7b
MODEL=/mnt/petrelfs/share_data/quxiaoye/models/llama2_7B

# run AWQ search (optional; we provided the pre-computed results)
# python -m awq.entry 
source activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/llm-awq-main

python awq/entry.py \
    --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq ../awq_cache/$MODEL-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python awq.entry \
    --model_path $MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq ../awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake

# generate real quantized weights (w4)
python awq.entry \
    --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq ../awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant ../quant_cache/$MODEL-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
python awq.entry \
    --model_path $MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant ../quant_cache/$MODEL-w4-g128-awq.pt