# # sim128 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-128samples.pt")
# # sim256 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-256samples.pt")
# # sim512 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-512samples.pt")
# # sim1024 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-1024samples.pt")
#
# sim128 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/Mixtral-block-c4_train-128samples.pt")
# sim256 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/Mixtral-block-c4_train-256samples.pt")
# sim512 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/Mixtral-block-c4_train-512samples.pt")
# sim1024 = torch.load("D:/Codes/20240223_moe_compression/results_prune/cache/DeepSeek-block-c4_train-1024samples.pt")
#
# _, indices128 = torch.sort(sim128[:, 0])
# _, indices256 = torch.sort(sim256[:, 0])
# _, indices512 = torch.sort(sim512[:, 0])
# _, indices1024 = torch.sort(sim1024[:, 0])
#
# print("indices128", indices128.tolist())
# print("indices256", indices256.tolist())
# print("indices512", indices512.tolist())
# print("indices1024", indices1024.tolist())
################################################################################################################
import json


def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path, indent=4, **kwargs):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


# file = "/mnt/petrelfs/dongdaize.d/workspace/compression/data/MetaMathQA-395K.json"
file = r"D:\Codes\20240223_moe_compression\data\MetaMathQA-395K.json"

data = load_json(file)
save_json(data, file, indent=2)

print("Done!")
