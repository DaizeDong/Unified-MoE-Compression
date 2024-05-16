import sys

import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
from tqdm import tqdm

sys.path = ["/mnt/petrelfs/dongdaize.d/workspace/compression/src"] + sys.path
SparseSemiStructuredTensor._FORCE_CUTLASS = True

from transformers import AutoConfig
from transformers.models.mixtral.modeling_mixtral import is_semi_structured_weight, MixtralSparseMoeBlock
from transformers.models.pruning_modules import ExpertLinear, GateLinear


##############################################################################

def convert_semi_structured_weights(model, tolerance_rate: float = 5e-7):
    # 🔍 Automatically check & convert semi-structured sparse weights in the model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, ExpertLinear, GateLinear)):
            if is_semi_structured_weight(module.weight.data, tolerance_rate=tolerance_rate):
                module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))
                if module.bias is None:  # we need to add bias to the model otherwise there will be error
                    module.bias = torch.nn.Parameter(torch.zeros(module.out_features, dtype=module.weight.dtype, device=module.weight.device))
                print(f"Converted {name} weights to semi-structured weights")
    return model


model_path = "/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1"
config = AutoConfig.from_pretrained(model_path)
config.num_local_experts = 32
prune_n, prune_m = 2, 4
x = torch.rand(256, 2048, config.hidden_size).bfloat16().cuda()

# mlp = nn.Linear(config.hidden_size, 14336).bfloat16().cuda().eval()
# mlp = nn.Linear(config.hidden_size, 14336, bias=False).bfloat16().cuda().eval()
mlp = MixtralSparseMoeBlock(config, 0, "static").bfloat16().cuda().eval()

# for name, module in mlp.named_modules():
#     if isinstance(module, (nn.Linear, ExpertLinear, GateLinear)):
#         if module.bias is None:  # we need to add bias to the model otherwise there will be error
#             module.bias = torch.nn.Parameter(torch.zeros(module.out_features, dtype=module.weight.dtype, device=module.weight.device))

# dense inference.
with torch.inference_mode():
    ##################################
    print("Dense:")
    x = x.cpu()
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated() / (1024 ** 2)
    x = x.cuda()
    print(f"Mem: {mem:.3f}MB")
    # dense_output = mlp(x)
    dense_t = sum([Timer(
        stmt="mlp(x)",
        globals={
            "mlp": mlp,
            "x": x
        }
    ).blocked_autorange().median * 1e3 for i in range(20)])
    print(dense_t)

    ##################################
    for name, param in tqdm(mlp.named_parameters()):
        if "gate" not in name and param.data.dim() >= 2:
            # 🔍 semi-structured n:m sparsity
            W_mask = torch.zeros_like(param.data, dtype=torch.float32).cuda()  # initialize a mask to be all 0
            for ii in range(param.data.shape[1]):
                if ii % prune_m == 0:
                    tmp = param.data[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

            # 🔍 Mask weights
            param.data = (param.data.float() * W_mask).bfloat16()
            # print(name,W_mask.sum(), (param.data == 0).sum(), is_semi_structured_weight(param))

    mlp = convert_semi_structured_weights(mlp)

    # F.linear(x, mlp.weight, None)

    ##################################
    print("Sparse:")
    x = x.cpu()
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated() / (1024 ** 2)
    x = x.cuda()
    print(f"Mem: {mem:.3f}MB")
    # sparse_output = mlp(x)
    sparse_t = sum([Timer(
        stmt="mlp(x)",
        globals={
            "mlp": mlp,
            "x": x
        }
    ).blocked_autorange().median * 1e3 for i in range(20)])
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
