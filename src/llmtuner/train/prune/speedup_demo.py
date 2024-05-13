import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
import sys
from tqdm import tqdm

sys.path = ["/mnt/petrelfs/dongdaize.d/workspace/compression/src"] + sys.path
SparseSemiStructuredTensor._FORCE_CUTLASS = True

from transformers import AutoConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

model_path = "/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1"
config = AutoConfig.from_pretrained(model_path)

layer_index = range(config.num_local_experts)
mode = "static"
mlp = MixtralSparseMoeBlock(config, layer_index, mode).half().cuda().eval()
prune_n, prune_m = 2, 4
x = torch.rand(64, 2048, config.hidden_size).half().cuda()

# dense inference. 
with torch.inference_mode():
    dense_output = mlp(x)
    dense_t = Timer(stmt="mlp(x)",
                    globals={"mlp": mlp,
                             "x": x}).blocked_autorange().median * 1e3
    print(dense_t)
    for name, param in tqdm(mlp.named_parameters()):
        if "gate" not in name:
            W_mask = torch.zeros_like(param.data).cuda()  # initialize a mask to be all 0
            if prune_n != 0:
                # 🔍 semi-structured n:m sparsity
                for ii in range(param.data.shape[1]):
                    if ii % prune_m == 0:
                        tmp = param.data[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                        # param.data = param.data * (torch.ones_like(W_mask) - W_mask)
                param = torch.nn.Parameter(to_sparse_semi_structured(param * W_mask.bool()))

    mem = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Mem: {mem:.3f}MB")
    sparse_output = mlp(x)
    sparse_t = Timer(stmt="mlp(x)",
                    globals={"mlp": mlp,
                              "x": x}).blocked_autorange().median * 1e3
    print(sparse_t)
    # sparse and dense matmul are numerically equivalent
    # assert torch.allclose(sparse_output, dense_output, atol=1e-3)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")



# # mask Linear weight to be 2:4 sparse
# # mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
# # linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
# # linear.weight = torch.nn.Parameter(mask * linear.weight)

# # mask Linear weight to be 4:8 sparse
# mask = torch.Tensor([0, 0, 1, 0, 1, 0, 1, 1]).tile((3072, 1280)).cuda().bool()
# linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
# linear.weight = torch.nn.Parameter(mask * linear.weight)

# print(mask)

# x = torch.rand(3072, 10240).half().cuda()

# with torch.inference_mode():
#     dense_output = linear(x)
#     dense_t = Timer(stmt="linear(x)",
#                     globals={"linear": linear,
#                              "x": x}).blocked_autorange().median * 1e3
#     print(dense_t)
    