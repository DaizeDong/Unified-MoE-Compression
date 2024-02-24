#!/usr/bin/bash

#SBATCH --job-name=ttt
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_test/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_test/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved

# reserved spot
num_nodes=2        # should match with --nodes
num_gpu_per_node=8 # should match with --gres
export OMP_NUM_THREADS=8
export LOGLEVEL=INFO

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"
echo "Node list: $SLURM_JOB_NODELIS"

srun torchrun \
  --nnodes ${num_nodes} \
  --nproc_per_node ${num_gpu_per_node} \
  --node_rank $SLURM_NODEID \
  src/train_bash.py \
  --stage pt \
  --do_eval \
  --model_name_or_path path_to_llama_model \
  --dataset lima \
  --finetuning_type full \
  --output_dir path_to_pt_checkpoint \
  --overwrite_cache \
  --per_device_train_batch_size 4 \
  --logging_steps 10 \
  --plot_loss \
  --fp16
