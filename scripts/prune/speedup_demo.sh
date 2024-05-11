#!/usr/bin/bash

#SBATCH --job-name=speedup
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_quantization/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/logs_quantization/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
# reserved spot auto

#####################################################################################################################
source ~/anaconda3/bin/activate compression
cd /mnt/petrelfs/dongdaize.d/workspace/compression

# OMP_NUM_THREADS=8 srun --partition=MoE --job-name=temp --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --kill-on-bad-exit=1 \
python src/llmtuner/train/prune/speedup_demo.py
