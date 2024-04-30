import torch
from safetensors.torch import load_file


ckpt = load_file("/mnt/petrelfs/dongdaize.d/workspace/compression/results_sft/Mixtral-8x7B-v0.1-bits4/checkpoint-50/adapter_model.safetensors", device="cpu")
# print(f"ckpt: {ckpt.keys()}")

for key in ckpt:
    if "gate" not in key:
        print(key, ckpt[key])