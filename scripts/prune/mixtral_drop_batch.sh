#####################################################################################################################
model_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_quantization

source ~/anaconda3/bin/activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression/scripts/prune

model_list=`ls $model_dir`
bits=4

# hyperparameters for expert drop. 
expert_drop_method="global_pruning" # layerwise_pruning global_pruning
r=6
# hyperparameters for layer drop. 
layer_drop_method="discrete"
drop_n_layer=8

# hyperparameters for block drop. 
block_drop_method="discrete"
drop_n_block=5

# echo $model_list
for file in $model_list
do 
    file=Mixtral-8x7B-v0.1-AWQ-4bits
    # file=deepseek-AWQ-4bits
    model_path=$model_dir/$file
    # quant_path=$quant_dir/$file-AWQ-${bits}bits/checkpoint
    # model=${model_path##*/}
    if [ -d "$model_path" ]; then
        echo $model_path
        # expert drop
        sbatch mixtral_expert_drop_quant.sh $model_path $expert_drop_method $r
        # layer drop
        # sbatch mixtral_layer_drop_quant.sh $model_path $layer_drop_method $drop_n_layer
        # block drop
        # sbatch mixtral_block_drop_quant.sh $model_path $block_drop_method $drop_n_block
        break
        sleep 1
    fi
done