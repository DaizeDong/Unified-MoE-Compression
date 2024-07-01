import torch
from torch import nn as nn, cuda

from llmtuner.model.deepseek.modeling_deepseek import DeepseekPreTrainedModel
from llmtuner.model.mixtral.modeling_mixtral import MixtralPreTrainedModel


def print_gpu_memory(accelerator):
    if accelerator.is_local_main_process:
        for i in range(cuda.device_count()):
            used_memory = cuda.memory_allocated(0) // 1024 ** 2
            print(f"GPU {i} Used Memory: {used_memory}MB")


def find_modules(module, layers=[], name='') -> dict:
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_moe_expert_linears(module, exclude_names: str = None) -> dict:
    # 🔍 find only the expert weights
    res = find_modules(module, [nn.Linear])
    for key in list(res.keys()):
        if "gate." in key:
            res.pop(key)
        if "self_attn." in key:
            res.pop(key)
    if exclude_names is not None:
        exclude_names = exclude_names.split(',')
        for module_name in list(res.keys()):
            for exclude_name in exclude_names:
                if exclude_name in module_name:
                    res.pop(module_name)
                    break
    return res


@torch.no_grad()
def check_sparsity_from_state_dict(state_dict):
    """
    🔍 This function has been rewritten to calculate sparsity from "state_dict".
    """
    # Get corresponding names for each layer
    layer_params = {}
    for name in sorted(list(state_dict.keys())):
        # Example (Mixtral): model.layers.5.block_sparse_moe.experts.2.w3.weight
        # Example (DeepSeek): model.layers.13.mlp.experts.28.up_proj
        # print(f"name: {name}")

        if "layers" in name:
            layer_id = int(name.split(".")[2])
            if layer_id not in layer_params:
                layer_params[layer_id] = [name]
            else:
                layer_params[layer_id].append(name)
        # print(f"layer_params: {layer_params}")
    layer_num = max(list(layer_params.keys())) + 1

    # Calculate sparsity
    count = 0
    total_params = 0
    for i in range(layer_num):
        if i in layer_params:
            sub_count = 0
            sub_params = 0
            for name in layer_params[i]:
                count += (state_dict[name] == 0).sum().item()
                total_params += state_dict[name].numel()

                sub_count += (state_dict[name] == 0).sum().item()
                sub_params += state_dict[name].numel()
            print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        else:
            print(f"layer {i} sparsity {0.0:.6f}")

    return float(count) / total_params


@torch.no_grad()
def prepare_calibration_input(model, dataloader, num_samples=16):
    layers = model.model.layers
    cache = {'inputs': [], 'attention_mask': [], "position_ids": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input, **kwargs):
            # print(input.shape)
            cache['inputs'].append(input)
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs['position_ids'])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for index, batch in enumerate(dataloader):
        if index >= num_samples:  # 🔍 limit the number of samples in each device, batch_size must be 1
            break
        try:
            model(**batch)
        except:
            pass
    layers[0] = layers[0].module

    outputs = [None] * len(cache['inputs'])

    return cache['inputs'], outputs, cache['attention_mask'], cache['position_ids']


def get_moe_model_information(model, accelerator=None):
    # 🔍 Get MoE layer ids
    if isinstance(model, MixtralPreTrainedModel):
        num_experts = model.config.num_local_experts
        num_layers = model.config.num_hidden_layers
        moe_layer_indices = list(range(num_layers))
    elif isinstance(model, DeepseekPreTrainedModel):
        num_experts = model.config.n_routed_experts
        num_layers = model.config.num_hidden_layers
        moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (model.config.n_routed_experts is not None and layer_idx >= model.config.first_k_dense_replace and layer_idx % model.config.moe_layer_freq == 0)]
    else:
        raise NotImplementedError

    # 🔍 Get valid MoE layer ids ("valid" denotes that this layer contains MoENorm & MoE)
    if isinstance(num_experts, list):
        valid_moe_layer_indices = [i for i in moe_layer_indices if num_experts[i] >= 0]
    else:
        valid_moe_layer_indices = moe_layer_indices

    # Print information
    if accelerator is None:
        print("num_experts", num_experts)
        print("num_layers", num_layers)
        print("moe_layer_indices", moe_layer_indices)
        print("valid_moe_layer_indices", valid_moe_layer_indices)
    else:
        accelerator.print("num_experts", num_experts)
        accelerator.print("num_layers", num_layers)
        accelerator.print("moe_layer_indices", moe_layer_indices)
        accelerator.print("valid_moe_layer_indices", valid_moe_layer_indices)

    return num_experts, num_layers, moe_layer_indices, valid_moe_layer_indices
