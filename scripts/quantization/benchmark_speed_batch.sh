#####################################################################################################################
source ~/anaconda3/bin/activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression

model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1
# model_path=/mnt/petrelfs/share_data/quxiaoye/models/Mistral-7B-v0.1
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek

# GPTQ
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mixtral-8x7B-v0.1-GPTQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mistral-7B-v0.1-GPTQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/deepseek-GPTQ-4bits/checkpoint

# AWQ
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/deepseek-AWQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mistral-7B-v0.1-AWQ-4bits/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/Mixtral-8x7B-v0.1-AWQ-4bits/checkpoint

# Combination
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/DeepSeek-expert_drop-global_pruning-r56-block_drop-discrete-drop4/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/Mixtral-block_drop-discrete-drop17/checkpoint

# AWQ + BLOCK
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/Mixtral-8x7B-v0.1-AWQ-4bits-block_drop-discrete-drop5
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/deepseek-AWQ-4bits-block_drop-discrete-drop5/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/deepseek-AWQ-4bits-block_drop-discrete-drop4/checkpoint

# AWQ + EXPERT
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/Mixtral-8x7B-v0.1-AWQ-4bits-expert_drop-global_pruning-r7/checkpoint
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/Mixtral-8x7B-v0.1-AWQ-4bits-expert_drop-global_pruning-r6/checkpoint
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/Mixtral-8x7B-v0.1-AWQ-4bits-expert_drop-global_pruning-r6
model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/Mixtral-8x7B-v0.1-AWQ-4bits-expert_drop-global_pruning-r7

# AWQ + LAYER
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble/Mixtral-8x7B-v0.1-AWQ-4bits-layer_drop-discrete-drop8/checkpoint

# AWQ
# model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization/deepseek-AWQ-4bits/checkpoint

save_file="/mnt/petrelfs/dongdaize.d/workspace/compression/results_speedup/xxxxxxxx.csv"
model_type="normal"


# echo $model_path

# python src/benchmark_speed.py \
#   --model_path $model_path \
#   --save_file ${save_file} \
#   # --model_type $model_type \
#   # --pretrained

model_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble
model_list=`ls $model_dir`

for file in $model_list
do 
    model_path=$model_dir/$file/checkpoint
    if [ -d "$model_path" ]; then
      echo $model_path
      sbatch /mnt/petrelfs/dongdaize.d/workspace/compression/scripts/quantization/benchmark_speed_single.sh $model_path
      break
      sleep 1
    # python src/benchmark_speed.py \
      # --model_path $model_path \
      # --save_file ${save_file} \
      # --model_type $model_type \
      # --pretrained
    fi
done
