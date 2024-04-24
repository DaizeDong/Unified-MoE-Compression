import torch
from torch import nn as nn, cuda

from transformers.models.mixtral.modeling_mixtral import ExpertLinear, MixtralSparseMoeBlock


def print_gpu_memory(accelerator):
    if accelerator.is_local_main_process:  # 🔍
        for i in range(cuda.device_count()):
            used_memory = cuda.memory_allocated(0) // 1024 ** 2
            print(f"GPU {i} Used Memory: {used_memory}MB")


def print_gpu_memory_device():
    device = cuda.current_device()
    used_memory = cuda.memory_allocated(device) // 1024 ** 2
    print(f"GPU {device} Used Memory: {used_memory}MB")


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
    # print(name, type(module), type(module) in layers)
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_moe_expert_linears_and_gate(module, layers=[nn.Linear, ExpertLinear], name='') -> dict:
    # 🔍 find the expert weights and gate weights
    res = find_modules(module, layers, name)
    for key in list(res.keys()):
        if 'self_attn' in key:
            res.pop(key)
    return res


def find_moe_expert_linears(module, layers=[nn.Linear, ExpertLinear], name='') -> dict:
    # 🔍 find only the expert weights
    res = find_modules(module, layers, name)
    for key in list(res.keys()):
        if "gate" in key:
            res.pop(key)
        if 'self_attn' in key:
            res.pop(key)
    return res


def find_moe_gates(module, layers=[nn.Linear], name='') -> dict:
    # 🔍 find only the gate network
    res = find_modules(module, layers, name)
    for key in list(res.keys()):
        if "gate" not in key:
            res.pop(key)
    return res


def find_moe_experts(module, layers=[MixtralSparseMoeBlock], name='') -> dict:
    # 🔍 find only the expert blocks
    res = find_modules(module, layers, name)
    return res


@torch.no_grad()
def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_modules(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


@torch.no_grad()
def check_sparsity_from_state_dict(state_dict):
    """
    🔍 This function has been rewritten to calculate sparsity from "state_dict".
    """
    # Get corresponding names for each layer
    layer_params = {}
    for name in sorted(list(state_dict.keys())):
        # Example: model.layers.5.block_sparse_moe.experts.2.w3.weight
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
        sub_count = 0
        sub_params = 0
        for name in layer_params[i]:
            count += (state_dict[name] == 0).sum().item()
            total_params += state_dict[name].numel()

            sub_count += (state_dict[name] == 0).sum().item()
            sub_params += state_dict[name].numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

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
        except ValueError:
            pass
    layers[0] = layers[0].module

    outputs = [None] * len(cache['inputs'])

    return cache['inputs'], outputs, cache['attention_mask'], cache['position_ids']
