#!/usr/bin/bash

#SBATCH --job-name=flops
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/calculate-flops.pytorch-main/test_examples/logs/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/calculate-flops.pytorch-main/test_examples/logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
# reserved spot auto

source activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/calculate-flops.pytorch-main/test_examples/

python test_llm.py
##############################################################################
