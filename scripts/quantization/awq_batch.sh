##############################################################################
# ${var##*/}
model_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune
quant_dir=/mnt/petrelfs/dongdaize.d/workspace/compression/results_assemble

model_list=`ls $model_dir`
# echo $model_list
source ~/anaconda3/bin/activate awq
cd /mnt/petrelfs/dongdaize.d/workspace/compression/scripts/quantization
bits=4

for file in $model_list
do 
    model_path=$model_dir/$file/checkpoint
    quant_path=$quant_dir/$file-AWQ-${bits}bits/checkpoint
    model=${model_path##*/}
    if [ -d "$model_path" ]; then
        echo $model_path
        echo $quant_path
        sbatch awq_single.sh $model_path $quant_path $bits
        break
        sleep 1
    fi
done