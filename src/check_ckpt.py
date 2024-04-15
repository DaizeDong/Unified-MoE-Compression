import torch

ckpt = torch.load('/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/Mixtral-wanda-c4_train-unstructured-0.5-128-NoAttn/checkpoint/model-00001-of-00019.safetensors')
print(ckpt.keys())