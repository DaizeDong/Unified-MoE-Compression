import torch

# sim128 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-128samples.pt")
# sim256 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-256samples.pt")
# sim512 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-512samples.pt")
# sim1024 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-1024samples.pt")

sim128 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/Mixtral-block-c4_train-128samples.pt")
sim256 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/Mixtral-block-c4_train-256samples.pt")
sim512 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/Mixtral-block-c4_train-512samples.pt")
sim1024 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-1024samples.pt")

_, indices128 = torch.sort(sim128[:, 0])
_, indices256 = torch.sort(sim256[:, 0])
_, indices512 = torch.sort(sim512[:, 0])
_, indices1024 = torch.sort(sim1024[:, 0])

print("indices128", indices128.tolist())
print("indices256", indices256.tolist())
print("indices512", indices512.tolist())
print("indices1024", indices1024.tolist())
