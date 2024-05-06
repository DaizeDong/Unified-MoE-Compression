import torch

from transformers import MixtralForCausalLM

model_old = MixtralForCausalLM.from_pretrained("/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1")
model_old_dict = model_old.state_dict()

model_new = MixtralForCausalLM.from_pretrained("/mnt/petrelfs/dongdaize.d/workspace/compression/results_prune/Mixtral-block_drop-consecutive-drop4/checkpoint")
model_new_dict = model_new.state_dict()

for key in sorted(list(model_old_dict.keys())):
    if key in model_new_dict:
        equal = torch.equal(model_old_dict[key], model_new_dict[key])
        print(key, equal)
