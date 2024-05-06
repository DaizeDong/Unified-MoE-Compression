import os
import sys

sys.path.append('/mnt/petrelfs/dongdaize.d/workspace/compression')
sys.path.append(os.getcwd())

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import is_deepspeed_zero3_enabled

from llmtuner.train.prune.utils import check_sparsity

model_name_or_path = "/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/test-Mixtral-wanda-unstructured-0.5/checkpoint/"
model_revision = "main"
hf_hub_token = None
cache_dir = None
compute_dtype = torch.bfloat16

config_kwargs = {
    "trust_remote_code": True,
    "cache_dir": cache_dir,
    "revision": model_revision,
    "token": hf_hub_token,
    "attn_implementation": "flash_attention_2",  # 🔍
}

config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    **config_kwargs,
)

check_sparsity(model)

# from safetensors import safe_open
# tensors = {}
# with safe_open(file_path, framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
#
# print(tensors.keys())
#
# for name, param in tensors.items():
#     sparse_count = (param == 0).sum().item()
#     total_count = param.numel()
#     print(f"{name}: {sparse_count}, {total_count}")
