from safetensors import safe_open

file_path = "/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/Mixtral-wanda-unstructured-0.5/checkpoint/model-00001-of-00019.safetensors"

tensors = {}
with safe_open(file_path, framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

print(tensors.keys())
print("Done!")
